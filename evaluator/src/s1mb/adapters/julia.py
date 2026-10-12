"""Julia's pinned marker encoder with strict native input budgets."""

import importlib
import json
import math

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, checkpoint_path


def julia_row(case, question):
    """Keep authored criteria and numeric rubric values without transport IDs."""
    definition = questions_for_api([question], structured=True)[question.id]

    def text(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    options = list(question.options)
    if question.task == "noul":
        options = sorted(options, key=lambda option: option.id == "true")
    labels = []
    for option in options:
        description = (
            json.loads(option.description_json)
            if option.description_json is not None
            else option.description
        )
        labels.append(
            text({"value": option.value, "description": description})
            if question.task == "score"
            else text(description)
        )
    return {
        "state": case.state,
        "question": text(definition["instructions"]),
        "type": question.task,
        "options": labels,
    }, [option.id for option in options]


class JuliaAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, device, context_limit=None):
        if not device.startswith("cuda"):
            raise ValueError("Julia requires explicit CUDA; no CPU fallback")
        if context_limit is not None and not 517 <= context_limit <= 8192:
            raise ValueError("Julia context_limit must be between 517 and 8192")
        path, resolved = checkpoint_path(model, revision)
        device = "cuda:0" if device == "cuda" else device
        self.setup("julia", model, resolved, str(path), device)
        runtime = importlib.import_module("julia.inference")
        self.native = importlib.import_module("julia.data")
        self.context_limit = context_limit or 8192
        self.engine = runtime.TransformerEngine(
            path, device=device, max_length=self.context_limit, head_length=512
        )
        self.attention_model = self.engine.model.encoder
        self.settings = {
            "dtype": "native-weights-bfloat16-autocast",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit,
            "head_length": 512,
            "max_candidates": 20,
            "max_option_tokens": 48,
            "input_length_policy": "reject-overflow-native-strict",
            "renderer": "native-markers-numeric-score-values-v1",
            "questions_per_call": 1,
            "source_revision": resolved,
            "backend": "TransformerEngine",
        }

    def predict(self, case):
        predictions = []
        for question in case.questions:
            row, keys = julia_row(case, question)
            self.native.validate_row(row, 1)
            row["_encoded"] = self.native.sequence(
                self.engine.tokenizer, row, self.context_limit, 512, strict=True
            )
            values = self.engine.logits([row])
            if len(values) != 1 or len(values[0]) != len(keys):
                raise ValueError("Julia returned incorrect candidate alignment")
            logits = values[0]
            if not all(math.isfinite(value) for value in logits):
                raise ValueError("Julia returned nonfinite logits")
            maximum = max(logits)
            masses = [math.exp(value - maximum) for value in logits]
            total = sum(masses)
            probabilities = dict(zip(keys, [value / total for value in masses], strict=True))
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
