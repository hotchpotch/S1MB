"""Complete token budgets, structured role records and native answer transport."""

import json
from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.llm2jev import (
    BoundedHF,
    LLM2JevAdapter,
    llm2jev_questions,
    llm2jev_state,
)


def test_structured_authored_noul_and_numeric_score():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested":true}'
    case.questions[2].options.reverse()
    questions = llm2jev_questions(case)
    assert questions["choose"]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["criteria"] == [
        {"value": 30.0, "description": "High"}, {"value": 10.0, "description": "Low"}
    ]
    assert "gold-must-not-be-sent" not in str(questions)


@pytest.mark.parametrize("messages", [
    [{"role": "customer", "message": "Refund"}],
    [{"role": "user", "content": "Refund", "source": "authored source"}],
    [{"role": {"nested": True}, "content": "Refund"}],
])
def test_role_records_keep_every_field(messages):
    state = {"messages": messages}
    assert json.loads(llm2jev_state(state)) == state


def test_actual_chat_remains_native():
    state = {"messages": [{"role": "user", "content": "Refund please"}]}
    assert llm2jev_state(state) is state


def test_overflow_precedes_gpu_scoring_and_boundary_is_accepted():
    calls = []
    backend = SimpleNamespace(
        tok=SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text)))),
        score=lambda *args: calls.append(args) or ([0.0], len(args[0])),
    )
    bounded = BoundedHF(backend, 3)
    assert bounded.score("abc", [], [10]) == ([0.0], 3)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="truncation"):
        bounded.score("abcd", [], [10])
    with pytest.raises(ValueError, match="text-only"):
        bounded.score("a", ["image"], [10])
    assert len(calls) == 1


def test_native_single_question_alignment():
    adapter = LLM2JevAdapter.__new__(LLM2JevAdapter)

    def engine(state, questions):
        assert state == cases().state
        assert set(questions) == {"decision"}
        question = questions["decision"]
        task = question["type"]
        if task == "noul":
            answer = {"type": task, "noul": 0.7}
        else:
            keys = list(question["criteria"]) if task == "choice" else ["0", "1"]
            answer = {"type": task, "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}
        return {"decision": answer}

    adapter.engine = engine
    result = adapter.predict(cases())
    assert result[0].probabilities == {"x": 0.2, "y": 0.8}
    assert result[1].probabilities["true"] == 0.7
    assert result[2].probabilities == {"low": 0.2, "high": 0.8}


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_before_loading(temperature):
    with pytest.raises(ValueError, match="temperature"):
        LLM2JevAdapter("unused", "main", "unused", "cuda:0", temperature=temperature)


def test_invalid_budget_before_loading():
    with pytest.raises(ValueError, match="positive"):
        LLM2JevAdapter("unused", "main", "unused", "cuda:0", temperature=1, context_limit=0)
