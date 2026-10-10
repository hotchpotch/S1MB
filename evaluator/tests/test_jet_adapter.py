"""Jet transport preserves structured definitions and numeric Score values."""

import json

from s1mb.adapters.jet import jet_questions
from s1mb.data import InferenceCase


def test_structured_score_values_survive_native_string_transport():
    case = InferenceCase.model_validate({
        "case_id": "sample", "state": "urgent",
        "questions": [{"id": "rate", "task": "score", "instructions": "Rate",
                       "options": [{"id": "low", "value": 10, "description": "Later"},
                                   {"id": "high", "value": 40, "description": "Today"}]}],
    })
    question = jet_questions(case)["rate"]
    assert question["type"] == "choice"
    assert json.loads(question["criteria"]["option_1"]) == {
        "value": 40.0, "description": "Today"}


def test_readout_preserves_unrounded_native_distribution(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    class NativeLogits:
        def double(self):
            return self

        def __truediv__(self, temperature):
            assert temperature == 1.0
            return self

        def softmax(self, dimension):
            assert dimension == -1
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [1 / 3] * 3

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))

    from s1mb.adapters.jet import JetAdapter

    case = InferenceCase.model_validate({
        "case_id": "sample", "state": "message",
        "questions": [{"id": "pick", "task": "choice", "instructions": "Select",
                       "options": [{"id": str(i), "description": str(i)} for i in range(3)]}],
    })
    adapter = object.__new__(JetAdapter)
    adapter.engine = SimpleNamespace(tokenizer=None, model=None, max_tokens=10,
                                     temperatures={"choice": 1.0})
    native_question = SimpleNamespace(type="choice", keys=[f"option_{i}" for i in range(3)])
    adapter.native_format = SimpleNamespace(
        Question=SimpleNamespace(from_dict=lambda value: native_question),
        label_token_ids=lambda tokenizer, question: [1, 2, 3],
    )
    adapter.native_inference = SimpleNamespace(encode=lambda *args: [1, 2])
    adapter.native_runtime = SimpleNamespace(label_logits=lambda *args: NativeLogits())
    probabilities = adapter.predict(case)[0].probabilities
    assert sum(round(value, 4) for value in probabilities.values()) == 0.9999
    assert abs(sum(probabilities.values()) - 1.0) < 1e-12
    assert all(abs(value - 1 / 3) < 1e-12 for value in probabilities.values())
