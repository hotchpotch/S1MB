"""Lev1 authored criteria and native distribution alignment."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.lev1 import Lev1Adapter


def test_native_mapping_keeps_authored_noul_and_nonuniform_score():
    adapter = Lev1Adapter.__new__(Lev1Adapter)
    seen = []

    def score(state, questions):
        assert state == cases().state
        assert set(questions) == {"decision"}
        question = questions["decision"]
        assert question["type"] == "choice"
        seen.append(question)
        return {"decision": dict(zip(question["criteria"], [0.2, 0.8], strict=True))}

    adapter.engine = SimpleNamespace(score=score)
    result = adapter.predict(cases())
    assert seen[1]["criteria"] == {"true": "Yes", "false": "No"}
    assert seen[2]["criteria"] == {
        "option_0": {"value": 10.0, "description": "Low"},
        "option_1": {"value": 30.0, "description": "High"},
    }
    assert "gold-must-not-be-sent" not in str(seen)
    assert result[0].probabilities == {"x": 0.2, "y": 0.8}
    assert result[1].probabilities == {"true": 0.2, "false": 0.8}
    assert result[2].probabilities == {"low": 0.2, "high": 0.8}


def test_unexpected_native_question_count_rejected():
    adapter = Lev1Adapter.__new__(Lev1Adapter)
    adapter.engine = SimpleNamespace(score=lambda *args: {"other": {}})
    with pytest.raises(ValueError, match="question count"):
        adapter.predict(cases())


def test_invalid_budget_before_load():
    with pytest.raises(ValueError, match="positive"):
        Lev1Adapter("unused", "main", "unused", "cuda:0", context_limit=0)
