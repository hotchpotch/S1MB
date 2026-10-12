"""Pinned JevK5-Lite classification with native calibration and overflow rejection."""

import importlib
import json
import logging
import math

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .input_budget import budget_state
from .upstream import UpstreamAdapter, state_text


def lite_inputs(state, question, compact_labels=False):
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
    if compact_labels and question.task != "score":
        labels = [state_text(definition) for definition in definitions]
        if len(set(labels)) != len(labels):
            labels = [f"{i}: {label}" for i, label in enumerate(labels)]
    return state_text(state), {instructions: labels}, labels


def budget_lite_head(engine, instructions, labels, context_limit):
    """Fit a native single head while reserving its mandatory 16 body tokens."""
    if context_limit <= 19:
        raise ValueError("JevK5-Lite fixed schema exceeds context limit; refusing truncation")
    def measure(state):
        task = state["question"].replace("_", " ")
        return 1 + len(engine._piece(f"{task} (one)")) + sum(
            1 + len(engine._piece(label)) for label in state["body"]
        )
    result = budget_state(
        {"question": instructions, "body": labels}, engine.tok,
        lambda state: json.dumps(state, ensure_ascii=False), measure,
        context_limit - 19, 1.0, min_retained_tokens=16,
    )
    return {instructions: result.state["body"]}, result.state["body"], result


class JevK5LiteAdapter(UpstreamAdapter):
    case_batch_size = 1
    native_adaptive_budget = False

    def __init__(self, model, revision, source, device, context_limit=None, native_adaptive_budget=False):
        self.native_adaptive_budget = native_adaptive_budget
        if context_limit is not None and not 1 <= context_limit <= 512:
            raise ValueError("JevK5-Lite context_limit must be between 1 and 512")
        self.setup("jevk5-lite", model, revision, source, device)
        self.native = importlib.import_module("jevk5.lite")
        self.engine = self.native.JevK5Lite.from_pretrained(
            str(self.path), device=device, dtype=self.torch.float32
        )
        self.context_limit = min(self.engine.max_len, context_limit or 512)
        if self.native_adaptive_budget:
            self.engine.max_len = self.context_limit
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

        if self.native_adaptive_budget:
            self.settings.update({
                "input_length_policy": "explicit-native-adaptive-budget-v2",
                "native_truncation_enabled": True,
                "head_budget_policy": "compact-label-prefix-on-overflow;head-tail-long-label-text;reserve-native-body-floor-16",
                "candidate_budget_policy": "all-options-in-one-native-head;preserve-option-order-and-count",
                "head_budget_score_policy": "preserve-complete-numeric-score-labels",
                "head_budget_min_retained_shortened_label_tokens": 16,
            })

    def predict(self, case):
        predictions = []
        for question in case.questions:
            text, tasks, labels = lite_inputs(case.state, question)
            heads = self.native._heads(tasks)
            ids, _, _, (start, end) = self.engine.encode(text, heads)
            # The native encoder shortens the body. Compare against its complete
            # tokenization before calling classify; a shortened input never reaches GPU.
            body_tokens = len(self.engine._piece(text))
            if len(ids) > self.context_limit:
                if not self.native_adaptive_budget or question.task == "score":
                    raise ValueError("JevK5-Lite fixed schema exceeds context limit; refusing truncation")
                head_before = len(ids) - (end - start) - 3
                text, tasks, labels = lite_inputs(case.state, question, compact_labels=True)
                instruction = next(iter(tasks))
                tasks, labels, budget = budget_lite_head(
                    self.engine, instruction, labels, self.context_limit,
                )
                heads = self.native._heads(tasks)
                ids, _, _, (start, end) = self.engine.encode(text, heads)
                if len(ids) > self.context_limit:
                    raise ValueError("JevK5-Lite native head still exceeds context limit")
                logging.getLogger(__name__).warning(
                    "Explicit head budget: case=%s question=%s schema=%d->%d candidates=%d policy=compact-head-tail-v2",
                    case.case_id, question.id, head_before, budget.request_after, len(labels),
                )
            if end - start != body_tokens:
                if not self.native_adaptive_budget:
                    raise ValueError("JevK5-Lite input exceeds context limit; refusing truncation")
                logging.getLogger(__name__).warning(
                    "Explicit native budget: case=%s question=%s adapter=jevk5-lite body=%d->%d candidates=%d",
                    case.case_id, question.id, body_tokens, end-start, len(labels))
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
