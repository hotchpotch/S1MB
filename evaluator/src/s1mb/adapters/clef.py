"""Pinned Cloudflare Clef native joint-schema inference without truncation."""

import importlib.util
import sys

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, checkpoint_path


def clef_record(case):
    """Anonymize transport IDs while preserving authored values and option order."""
    questions, mappings = {}, {}
    for index, question in enumerate(case.questions):
        key = f"field_{index:06d}"
        item = questions_for_api([question], structured=True)[question.id]
        if question.task == "choice":
            descriptions = list(item["criteria"].values())
            item["criteria"] = {
                f"option_{i:06d}": description for i, description in enumerate(descriptions)
            }
            mappings[key] = {
                f"option_{i:06d}": option.id for i, option in enumerate(question.options)
            }
        elif question.task == "score":
            item["criteria"] = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, item["criteria"], strict=True)
            ]
            mappings[key] = {str(i): option.id for i, option in enumerate(question.options)}
        else:
            mappings[key] = {option.id: option.id for option in question.options}
        questions[key] = item
    return {"state": case.state, "questions": questions}, mappings


class ClefAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("context_limit must be positive")
        if not device.startswith("cuda"):
            raise ValueError("Clef requires explicit CUDA; no CPU fallback")
        device = "cuda:0" if device == "cuda" else device
        path, resolved = checkpoint_path(model, revision)
        self.setup("clef", model, resolved, str(path), device)
        spec = importlib.util.spec_from_file_location(
            f"s1mb_clef_{resolved}", path / "joint_schema_model.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load pinned Clef runtime")
        self.native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.native
        spec.loader.exec_module(self.native)
        self.engine, self.processor = self.native.load_release_model(
            path, device=device, dtype=self.torch.bfloat16, attn_implementation="sdpa"
        )
        self.attention_model = self.engine.language_model
        self.context_limit = context_limit or 16384
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "questions_per_call": "all-case-questions",
            "renderer": "native-anonymous-ids-numeric-score-v1",
            "source_revision": resolved,
        }
        self.enable_kernels()

    def predict(self, case):
        record, mappings = clef_record(case)
        # The native encoder slices state to max_length. Encode without a practical
        # cap first, then reject before transferring any inputs to the GPU.
        encoded = self.native.encode_record(
            self.processor.tokenizer, record, max_length=sys.maxsize
        )
        if len(encoded.input_ids) > self.context_limit:
            raise ValueError(f"Clef input exceeds {self.context_limit} tokens; refusing truncation")
        if [q.question_id for q in encoded.questions] != list(record["questions"]):
            raise ValueError("Native question alignment differs from request")
        batch = self.native.collate_records(
            [encoded], self.processor.tokenizer.pad_token_id, self.torch.device(self.device)
        )
        with self.torch.inference_mode():
            logits = self.engine(batch)[0]
        predictions = []
        for question, native_question, values in zip(
            case.questions, encoded.questions, logits, strict=True
        ):
            mapping = mappings[native_question.question_id]
            if set(native_question.option_ids) != set(mapping):
                raise ValueError("Native option IDs differ from request")
            probabilities = dict(
                zip(
                    [mapping[key] for key in native_question.option_ids],
                    values.float().softmax(-1).cpu().tolist(),
                    strict=True,
                )
            )
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
