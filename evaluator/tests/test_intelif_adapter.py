"""Intelif anonymous transport preserves declared numeric values and IDs."""

from types import SimpleNamespace

from s1mb.adapters.intelif import IntelifAdapter
from s1mb.data import InferenceCase


def test_native_choice_transport_preserves_numeric_levels_and_alignment():
    case = InferenceCase.model_validate({
        "case_id": "sample", "state": {"message": "urgent"},
        "questions": [{"id": "rating", "task": "score", "instructions": "Rate urgency",
                       "options": [{"id": "low", "value": 10, "description": "Later"},
                                   {"id": "high", "value": 40, "description": "Today"}]}],
    })
    seen = []

    class Engine:
        def system_one(self, state, questions):
            seen.append(questions["decision"])
            return SimpleNamespace(answers={"decision": SimpleNamespace(
                probabilities={"option_0": 0.2, "option_1": 0.8})})

    adapter = IntelifAdapter.__new__(IntelifAdapter)
    adapter.engine = Engine()
    result = adapter.predict(case)[0]
    assert result.probabilities == {"low": 0.2, "high": 0.8}
    assert seen[0]["criteria"]["option_1"] == {"value": 40.0, "description": "Today"}
