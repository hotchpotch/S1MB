"""EXAONE native typed mapping, inference boundaries and module isolation."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.exaone_jev import ExaoneJevAdapter, exaone_questions


def test_preserves_authored_structures_and_score_levels():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested":true}'
    case.questions[2].options.reverse()
    questions = exaone_questions(case)
    assert questions["choose"]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["criteria"] == [
        {"value": 30.0, "description": "High"}, {"value": 10.0, "description": "Low"},
    ]
    assert "gold-must-not-be-sent" not in str(questions)


def test_native_response_alignment():
    adapter = ExaoneJevAdapter.__new__(ExaoneJevAdapter)

    async def ask(state, question):
        assert state == cases().state
        task = question["type"]
        if task == "noul":
            return {"type": task, "noul": 0.7}, 100
        keys = list(question["criteria"]) if task == "choice" else ["0", "1"]
        return {"type": task, "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}, 100

    adapter.native = SimpleNamespace(ask=ask)
    predictions = adapter.predict(cases())
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities["true"] == 0.7
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}


def test_transport_overflow_rejects_before_tensor_allocation():
    import asyncio

    adapter = ExaoneJevAdapter.__new__(ExaoneJevAdapter)
    adapter.context_limit = 10
    with pytest.raises(ValueError, match="refusing truncation"):
        asyncio.run(adapter._read(list(range(10)), 2))


def test_invalid_budget_before_load():
    with pytest.raises(ValueError, match="positive"):
        ExaoneJevAdapter("unused", "main", "unused", "cuda:0", context_limit=0)
