"""RSI native calibrated server API with explicit complete-input encoding."""

import dataclasses
import importlib
import os

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


def rsi_state_text(state, wire):
    """Treat unsupported chat-role objects as authored JSON data, preserving all fields."""
    try:
        return wire.state_to_text(state)
    except wire.RequestError as error:
        if str(error) != "Chat state has an unsupported message role":
            raise
        return wire.serialize(state)


class RSIAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        limit = 32768 if context_limit is None else context_limit
        if limit < 1:
            raise ValueError("RSI context_limit must be positive")
        self.setup("rsi", model, revision, source, device)
        native = importlib.import_module("serve.decider")
        self.wire = importlib.import_module("serve.wire")
        overrides = {"RSIJEV_MAX_LENGTH": "", "RSIJEV_TRUNCATE": ""}
        previous = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            self.engine = native.Decider(str(self.path), device=device, dtype="bf16", batch_size=1)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        served = self.engine.served
        served.enc = dataclasses.replace(served.enc, max_length=limit, truncate="none")
        served.meta["serving"] = {**served.meta.get("serving", {}),
                                  "max_length": limit, "truncate": "none"}
        if served.enc.truncate != "none" or served.enc.max_length != limit:
            raise ValueError("RSI did not activate complete-input rejection")
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "native-server",
            "max_input_tokens": limit, "input_length_policy": "reject-overflow-native",
            "questions_per_call": 1, "case_batch_size": 1,
            "calibration": self.engine.calibration,
            "native_spec": served.meta["spec"],
            "runtime": "native-in-process-server-calibrated-option-readout",
            "renderer": "native-choice-authored-noul-numeric-score",
            "structured_state_policy": "native-text;unsupported-chat-roles-as-complete-json-data",
        })

    def predict(self, case):
        predictions = []
        questions = sifr_questions(case)
        for question in case.questions:
            response = self.engine.request(rsi_state_text(case.state, self.wire), {"decision": questions[question.id]})
            if response.get("truncated") is not None:
                raise ValueError("RSI reported unexpected input truncation")
            if set(response["answers"]) != {"decision"}:
                raise ValueError("RSI returned incorrect answer count")
            probabilities = response["answers"]["decision"]["probabilities"]
            keys = ([o.id for o in question.options] if question.task == "noul"
                    else [f"option_{i}" for i in range(len(question.options))])
            check_probabilities(probabilities, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={o.id: probabilities[key] for o, key in
                               zip(question.options, keys, strict=True)},
            ))
        return predictions
