"""Kas transport retains criteria and rejects unsupported candidate counts."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.kas import KasAdapter
from s1mb.data import InferenceCase


def test_numeric_score_transport_and_original_probability_alignment():
    case = InferenceCase.model_validate({
        "case_id": "sample", "state": {"message": "urgent"},
        "questions": [{"id": "rating", "task": "score", "instructions": "Rate urgency",
                       "options": [{"id": "low", "value": 10, "description": "Later"},
                                   {"id": "high", "value": 40, "description": "Today"}]}],
    })
    seen = []

    class Engine:
        _labels = ("A", "B")

        def __call__(self, state, questions):
            seen.append(questions["decision"])
            return {"answers": {"decision": {
                "probabilities": {"option_0": 0.2, "option_1": 0.8}}}}, {}

    adapter = KasAdapter.__new__(KasAdapter)
    adapter.engine = Engine()
    result = adapter.predict(case)[0]
    assert result.probabilities == {"low": 0.2, "high": 0.8}
    assert seen[0]["criteria"]["option_1"] == {"value": 40.0, "description": "Today"}


def test_candidate_capacity_rejects_before_native_forward():
    case = InferenceCase.model_validate({
        "case_id": "sample", "state": "state",
        "questions": [{"id": "pick", "task": "choice", "instructions": "Choose",
                       "options": [{"id": str(i), "description": str(i)} for i in range(3)]}],
    })
    adapter = KasAdapter.__new__(KasAdapter)
    adapter.engine = SimpleNamespace(_labels=["A", "B"])
    with pytest.raises(ValueError, match="candidate count"):
        adapter.predict(case)
