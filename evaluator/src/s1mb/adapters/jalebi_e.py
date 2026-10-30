"""Jalebi-E (Airavat AI): a 66.6M-parameter decision encoder trained from scratch, scored in one pass per question."""

import gc
import importlib
import json

from s1mb.data import InferenceCase, ModelInfo, Prediction, check_probabilities
from s1mb.parameters import parameter_metadata

from .upstream import checkpoint_path


def record(case, question):
    """Plain text of one decision: the parsed state, the authored instructions and the option descriptions in declared order."""
    state = (
        case.state
        if isinstance(case.state, str)
        else json.dumps(case.state, ensure_ascii=False, sort_keys=True)
    )
    return state, question.instructions, [option.description for option in question.options]


class JalebiEAdapter:
    case_batch_size = 8
    batch_size = 32

    def __init__(self, model, revision, device, native_adaptive_budget=False, engine=None):
        """Load the pinned checkpoint. `engine` injects an already built jalebi_e_model.JalebiE (tests only)."""
        torch = importlib.import_module("torch")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
        self.torch, self.device = torch, device
        self.model_id = model
        if engine is None:
            path, self.revision = checkpoint_path(model, revision)
            engine = importlib.import_module("s1mb.adapters.jalebi_e_model").JalebiE.from_pretrained(
                str(path), device=device
            )
        else:
            self.revision = revision
        self.engine = engine
        self.trim = bool(native_adaptive_budget)
        self.bf16 = device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] >= 8
        self.tally = {"questions": 0, "rejected": 0, "state_trimmed": 0}

    def predict(self, case: InferenceCase) -> list[Prediction]:
        return self.predict_batch([case])[0]

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        out = [[None] * len(case.questions) for case in cases]
        jobs = []
        for ci, case in enumerate(cases):
            for qi, question in enumerate(case.questions):
                self.tally["questions"] += 1
                state, instructions, options = record(case, question)
                try:
                    # Validate the complete input length first: an overflowing question is an error, never a guess.
                    dropped = self.engine.encode(state, instructions, options, trim=self.trim)[2]
                except ValueError as error:
                    self.tally["rejected"] += 1
                    out[ci][qi] = Prediction(
                        case_id=case.case_id, question_id=question.id, error=str(error)
                    )
                    continue
                self.tally["state_trimmed"] += dropped > 0
                jobs.append((ci, qi, (state, instructions, options)))
        if jobs:
            probabilities, _ = self.engine.probabilities(
                [request for _, _, request in jobs],
                trim=self.trim,
                batch_size=self.batch_size,
                bf16=self.bf16,
            )
            for (ci, qi, _), values in zip(jobs, probabilities, strict=True):
                case, question = cases[ci], cases[ci].questions[qi]
                ids = [option.id for option in question.options]
                answer = dict(zip(ids, values, strict=True))
                check_probabilities(answer, ids)
                out[ci][qi] = Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=answer
                )
        return out

    def metadata(self) -> ModelInfo:
        from .jalebi_e_model import MIN_STATE, OPTION_CAP, QUESTION_CAP, WINDOW

        return ModelInfo(
            **parameter_metadata(self.engine),
            id=self.model_id,
            adapter="jalebi-e",
            revision=self.revision,
            settings={
                "device": self.device,
                "dtype": "float32-weights-bfloat16-autocast" if self.bf16 else "float32",
                "renderer": "jalebi-e-plain-state-instructions-descriptions-v1",
                "max_input_tokens": WINDOW,
                "input_length_policy": "trim-state" if self.trim else "reject-overflow",
                "trim_question_cap": QUESTION_CAP if self.trim else None,
                "trim_option_cap": OPTION_CAP if self.trim else None,
                "trim_min_state_tokens": MIN_STATE if self.trim else None,
                "options_per_window": "all",
                "questions_per_call": "batched",
                "batch_size": self.batch_size,
                "versions": {"torch": self.torch.__version__},
                "run_counts": dict(self.tally),
            },
        )

    def close(self):
        if hasattr(self, "engine"):
            del self.engine
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
