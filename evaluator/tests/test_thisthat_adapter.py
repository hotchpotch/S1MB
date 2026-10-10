"""Input preservation and native answer alignment for this-that."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.thisthat import ThisThatAdapter, native_questions


def request(text, options):
    return SimpleNamespace(text=text, options=tuple(options))


def test_authored_criteria_order_and_numeric_levels():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested":true}'
    case.questions[2].options.reverse()
    native = native_questions(case, request)
    assert json.loads(native[0].options[0]) == {"nested": True}
    assert native[1].options == ("Yes", "No")
    assert [json.loads(value) for value in native[2].options] == [
        {"value": 30.0, "description": "High"},
        {"value": 10.0, "description": "Low"},
    ]
    assert "gold-must-not-be-sent" not in str(native)


def test_duplicate_descriptions_keep_all_candidates_without_original_ids():
    native = native_questions(cases(), request)
    assert native[0].options == ("option_0: Same", "option_1: Same")


def adapter_with_prompt(length, calls):
    adapter = ThisThatAdapter.__new__(ThisThatAdapter)
    adapter.native = SimpleNamespace(Question=request)
    adapter.context_limit = 128
    adapter.tokenizer = SimpleNamespace(encode=lambda text, **kw: list(range(2000)))

    def build(tokenizer, state, questions, **kw):
        assert kw["max_state_tokens"] == 2000
        return {"ids": list(range(length))}

    def decide(state, question, **kw):
        calls.append(kw)
        return SimpleNamespace(options=question.options, probabilities=(0.2, 0.8))

    adapter.prompt = cast(Any, SimpleNamespace(build=build))
    adapter.engine = SimpleNamespace(decide=decide)
    return adapter


def test_full_state_budget_and_exact_padded_boundary():
    calls = []
    predictions = adapter_with_prompt(128, calls).predict(cases())
    assert len(calls) == 3
    assert all(call["max_state_tokens"] == 2000 for call in calls)
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities == {"true": 0.2, "false": 0.8}
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}


def test_padded_overflow_rejected_before_gpu_forward():
    calls = []
    with pytest.raises(ValueError, match="192 padded tokens.*refusing truncation"):
        adapter_with_prompt(129, calls).predict(cases())
    assert calls == []


def test_invalid_budget_before_load():
    with pytest.raises(ValueError, match="positive"):
        ThisThatAdapter("unused", "main", "unused", "cuda:0", context_limit=0)
