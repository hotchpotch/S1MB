"""OneJev transport preserves authored values and native answer alignment."""

from types import ModuleType, SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.onejev import OneJevAdapter, onejev_questions


def test_structured_criteria_numeric_score_and_no_identifiers():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    case.questions[2].options.reverse()
    questions = onejev_questions(case)
    assert questions["choose"]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["criteria"] == [
        {"value": 30.0, "description": "High"},
        {"value": 10.0, "description": "Low"},
    ]
    assert "gold-must-not-be-sent" not in str(questions)


def test_single_native_request_alignment_and_no_media():
    adapter = OneJevAdapter.__new__(OneJevAdapter)
    adapter.schema = ModuleType("synthetic_qev_schema")
    adapter.schema.__dict__["SystemOneRequest"] = SimpleNamespace(
        model_validate=lambda request: request
    )

    def decide(request, *, debias, media):
        assert debias == 1 and media == []
        assert request["state"] == cases().state
        assert set(request["questions"]) == {"decision"}
        question = request["questions"]["decision"]
        task = question["type"]
        if task == "noul":
            answer = {"type": task, "noul": 0.7}
        else:
            keys = list(question["criteria"]) if task == "choice" else ["0", "1"]
            answer = {"type": task, "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}
        return SimpleNamespace(model_dump=lambda **kwargs: {"answers": {"decision": answer}}), {}

    adapter.engine = SimpleNamespace(decide=decide)
    predictions = adapter.predict(cases())
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities["true"] == 0.7
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}


def test_invalid_budget_before_loading():
    with pytest.raises(ValueError, match="positive"):
        OneJevAdapter("unused", "main", "unused", "cuda:0", context_limit=0)
