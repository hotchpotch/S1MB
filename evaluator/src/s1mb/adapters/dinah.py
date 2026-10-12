"""Pinned Dinah Torch runtime with native lossless marker encoding."""

import importlib.util
import logging
import math
import sys

from .base import decode_answers, questions_for_api
from .input_budget import budget_state
from .upstream import UpstreamAdapter, checkpoint_path


def dinah_questions(case):
    """Preserve structured definitions, rubric values and anonymous Choice labels."""
    questions = questions_for_api(case.questions, structured=True)
    for question in case.questions:
        item = questions[question.id]
        if question.task == "choice":
            item["criteria"] = {
                f"option_{i}": value for i, value in enumerate(item["criteria"].values())
            }
        elif question.task == "score":
            item["criteria"] = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, item["criteria"], strict=True)
            ]
    return questions


class DinahAdapter(UpstreamAdapter):
    case_batch_size = 1
    state_token_reduction = 0.0

    def __init__(self, model, revision, device, context_limit=None, state_token_reduction=0.0):
        if not math.isfinite(state_token_reduction) or not 0 <= state_token_reduction <= 1:
            raise ValueError("State token reduction must be within [0, 1]")
        self.state_token_reduction = state_token_reduction
        if not device.startswith("cuda"):
            raise ValueError("Dinah requires explicit CUDA; no CPU fallback")
        if context_limit is not None and not 1 <= context_limit <= 8192:
            raise ValueError("Dinah context_limit must be between 1 and 8192")
        path, resolved = checkpoint_path(model, revision)
        device = "cuda:0" if device == "cuda" else device
        self.setup("dinah", model, resolved, str(path), device)
        spec = importlib.util.spec_from_file_location(f"s1mb_dinah_{resolved}", path / "dinah.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load pinned Dinah runtime")
        self.native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.native
        spec.loader.exec_module(self.native)
        self.engine = self.native.Dinah.from_pretrained(path, device=device, dtype="bfloat16")
        self.engine.max_length = min(self.engine.max_length, context_limit or 8192)
        self.attention_model = self.engine.net.encoder
        self.set_attention("sdpa")
        self.settings.update(
            {
                "dtype": "float32-weights-bfloat16-autocast",
                "max_input_tokens": self.engine.max_length,
                "input_length_policy": "reject-overflow-native",
                "renderer": "native-markers-structured-numeric-score-v1",
                "questions_per_call": 1,
                "source_revision": resolved,
                "backend": "torch",
            }
        )
        if self.state_token_reduction:
            self.settings.update(
                {
                    "input_length_policy": "explicit-state-head-tail-v1",
                    "max_state_token_reduction": self.state_token_reduction,
                    "state_budget_policy": "longest-unprotected-string-first;head-tail;json-structure-preserved",
                    "state_budget_min_string_tokens": 32,
                    "candidate_budget_policy": "preserve-all-options-and-authored-criteria",
                }
            )

    def budgeted_state(self, case, question):
        if not self.state_token_reduction:
            return case.state
        criteria = question["criteria"]
        options = (
            list(criteria.values())
            if question["type"] == "choice"
            else [criteria[key] for key in ["false", "true"]]
            if question["type"] == "noul"
            else criteria
        )
        render = self.native.render
        fixed = 3 + len(self.engine._ids(render(question["instructions"])))
        fixed += sum(1 + len(self.engine._ids(render(option))) for option in options)
        result = budget_state(
            case.state,
            self.engine.tokenizer,
            render,
            lambda state: fixed + len(self.engine._ids(render(state))),
            self.engine.max_length,
            self.state_token_reduction,
        )
        if result.request_before != result.request_after:
            logging.getLogger(__name__).warning(
                "Explicit state budget: case=%s request=%d->%d state=%d->%d policy=head-tail-v1",
                case.case_id,
                result.request_before,
                result.request_after,
                result.state_before,
                result.state_after,
            )
        return result.state

    def predict(self, case):
        answers = {}
        for key, question in dinah_questions(case).items():
            results = self.engine.predict(
                [dict(question, state=self.budgeted_state(case, question))], batch_size=1
            )
            if len(results) != 1:
                raise ValueError("Dinah returned incorrect answer count")
            answers[key] = results[0]
        return decode_answers(case, answers, anonymous_choice=True)
