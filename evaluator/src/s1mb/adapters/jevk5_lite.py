"""Pinned JevK5-Lite classification with native calibration and overflow rejection."""

import importlib
import math

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, state_text


def lite_inputs(state, question):
    """Render authored definitions without exposing dataset identifiers."""
    item = questions_for_api([question], structured=True)[question.id]
    instructions = state_text(item["instructions"])
    if question.task == "noul":
        definitions = [item["criteria"][option.id] for option in question.options]
    else:
        definitions = (
            list(item["criteria"].values()) if question.task == "choice" else item["criteria"]
        )
    labels = [
        f"level {i}: {option.value}: {state_text(definition)}"
        if question.task == "score"
        else f"option {i}: {state_text(definition)}"
        for i, (option, definition) in enumerate(zip(question.options, definitions, strict=True))
    ]
    return state_text(state), {instructions: labels}, labels


class JevK5LiteAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and not 1 <= context_limit <= 512:
            raise ValueError("JevK5-Lite context_limit must be between 1 and 512")
        self.setup("jevk5-lite", model, revision, source, device)
        self.native = importlib.import_module("jevk5.lite")
        self.engine = self.native.JevK5Lite.from_pretrained(
            str(self.path), device=device, dtype=self.torch.float32
        )
        self.context_limit = min(self.engine.max_len, context_limit or 512)
        if not math.isfinite(self.engine.temperature_single) or self.engine.temperature_single <= 0:
            raise ValueError("Invalid native single-label calibration temperature")
        self.settings.update(
            {
                "dtype": "float32",
                "max_input_tokens": self.context_limit,
                "input_length_policy": "reject-overflow",
                "attention_implementation": self.engine.encoder.config._attn_implementation,
                "probability_origin": "native-single-label-calibrated-softmax",
                "temperature_single": self.engine.temperature_single,
                "renderer": "native-head-structured-anonymous-numeric-score-v1",
            }
        )

    def predict(self, case):
        predictions = []
        for question in case.questions:
            text, tasks, labels = lite_inputs(case.state, question)
            heads = self.native._heads(tasks)
            ids, _, _, (start, end) = self.engine.encode(text, heads)
            # The native encoder shortens the body. Compare against its complete
            # tokenization before calling classify; a shortened input never reaches GPU.
            if len(ids) > self.context_limit or end - start != len(self.engine._piece(text)):
                raise ValueError("JevK5-Lite input exceeds context limit; refusing truncation")
            result = self.engine.classify(text, tasks)
            if set(result) != set(tasks):
                raise ValueError("JevK5-Lite returned incorrect heads")
            raw = result[next(iter(tasks))]["probabilities"]
            check_probabilities(raw, labels)
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=question.id,
                    probabilities={
                        option.id: raw[label]
                        for option, label in zip(question.options, labels, strict=True)
                    },
                )
            )
        return predictions
