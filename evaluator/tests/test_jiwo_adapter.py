"""Jiwo typed transport, option alignment and native Python requirement."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters import jiwo
from s1mb.adapters.jiwo import JiwoAdapter, jiwo_questions


def test_structured_anonymous_and_numeric_criteria():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested":true}'
    case.questions[2].options.reverse()
    questions = jiwo_questions(case)
    assert questions["choose"]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["criteria"] == [
        {"value": 30.0, "description": "High"}, {"value": 10.0, "description": "Low"}
    ]
    assert "gold-must-not-be-sent" not in str(questions)


def test_native_single_question_and_budget_alignment():
    adapter = JiwoAdapter.__new__(JiwoAdapter)
    adapter.context_limit = 32768

    def decide(state, questions, *, max_length, batch_tokens):
        assert state == cases().state
        assert max_length == batch_tokens == 32768
        assert set(questions) == {"decision"}
        question = questions["decision"]
        task = question["type"]
        if task == "noul":
            answer = {"type": task, "noul": 0.7}
        else:
            keys = list(question["criteria"]) if task == "choice" else ["0", "1"]
            answer = {"type": task, "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}
        return {"answers": {"decision": answer}}

    adapter.engine = SimpleNamespace(decide=decide)
    result = adapter.predict(cases())
    assert result[0].probabilities == {"x": 0.2, "y": 0.8}
    assert result[1].probabilities["true"] == 0.7
    assert result[2].probabilities == {"low": 0.2, "high": 0.8}


def test_native_wrong_answer_count_rejected():
    adapter = JiwoAdapter.__new__(JiwoAdapter)
    adapter.context_limit = 32768
    adapter.engine = SimpleNamespace(decide=lambda *args, **kwargs: {"answers": {"other": {}}})
    with pytest.raises(ValueError, match="count"):
        adapter.predict(cases())


def test_invalid_budget_rejected_before_model_load():
    with pytest.raises(ValueError, match="positive"):
        JiwoAdapter("unused", "main", "unused", "cuda:0", context_limit=0)


def test_native_python_requirement_is_explicit(monkeypatch):
    monkeypatch.setattr(jiwo.sys, "version_info", (3, 11, 0, "final", 0))
    with pytest.raises(RuntimeError, match="Python 3.12"):
        JiwoAdapter("unused", "main", "unused", "cuda:0")
