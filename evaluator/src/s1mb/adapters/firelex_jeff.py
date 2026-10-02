"""Firelex Jeff's pinned native backbone, decision readout and calibration."""

import importlib
import json
import sys

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter


def decision_row(state, question):
    """Preserve structured descriptions and numeric levels without exposing IDs."""
    item = questions_for_api([question], structured=True, anonymous_choice=True)[question.id]
    if question.task == "score":
        item["criteria"] = [
            {"value": option.value, "description": description}
            for option, description in zip(question.options, item["criteria"], strict=True)
        ]
    return {"state": state, "question": item}


class FirelexJeffAdapter(UpstreamAdapter):
    case_batch_size = 1
    max_candidates: int
    context_limit: int

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        if sys.version_info < (3, 12):
            raise RuntimeError("Firelex Jeff's upstream runtime requires Python 3.12 or later")
        if context_limit is not None and context_limit < 1:
            raise ValueError("context_limit must be positive")
        if max_candidates is not None and not 2 <= max_candidates <= 255:
            raise ValueError("max_candidates must be in 2..255")
        self.setup("firelex-jeff", model, revision, source, device)
        config = json.loads((self.path / "decision_config.json").read_text())
        native_limit = config.get("max_options")
        if not isinstance(native_limit, int) or not 2 <= native_limit <= 255:
            raise ValueError("Checkpoint must declare max_options in 2..255")
        self.max_candidates = max_candidates or native_limit
        self.context_limit = context_limit or 8192
        self.native = importlib.import_module("jeff.models")
        self.engine = self.native.load_decision_model(checkpoint=self.path, device=device).eval()
        self.attention_model = self.engine.backbone
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "temperature": self.engine.temperature,
            "input_length_policy": "reject-overflow",
            "max_input_tokens": self.context_limit,
            "max_candidates": self.max_candidates,
            "trained_max_candidates": native_limit,
            "case_batch_size": 1,
            "base_model": config["base_model"],
            "base_revision": config["revision"],
            "prompt_layout": config.get("prompt_layout", "state-first"),
            "renderer": "native-anonymous-choice-numeric-score-v1",
            "capacity_extension": {
                "context": self.context_limit > 8192,
                "candidates": self.max_candidates > native_limit,
            },
        }
        if config.get("architecture", "qwen") == "qwen":
            self.enable_kernels()

    def predict(self, case):
        predictions = []
        for question in case.questions:
            if len(question.options) > self.max_candidates:
                raise ValueError(
                    f"Firelex Jeff question exceeds {self.max_candidates} candidates; "
                    "refusing to drop options"
                )
            row = decision_row(case.state, question)
            batch = self.engine.prepare([row], max_length=self.context_limit)
            with self.torch.inference_mode():
                values = (self.engine(batch) / self.engine.temperature).softmax(-1)
                probabilities = values[0, : len(question.options)].float().cpu().tolist()
            ids = (
                ["false", "true"]
                if question.task == "noul"
                else [option.id for option in question.options]
            )
            mapped = dict(zip(ids, probabilities, strict=True))
            check_probabilities(mapped, [option.id for option in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=question.id,
                    probabilities=mapped,
                )
            )
        return predictions
