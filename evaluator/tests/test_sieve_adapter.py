"""Sieve's complete authored criteria and rejection before native trimming."""

import json
from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.sieve import SieveAdapter, sieve_questions


def test_structured_criteria_authored_noul_and_numeric_levels():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested":true}'
    case.questions[2].options.reverse()
    questions = sieve_questions(case)
    assert json.loads(questions[0][1][0]) == {"nested": True}
    assert questions[1][1] == ["Yes", "No"]
    assert [json.loads(value) for value in questions[2][1]] == [
        {"value": 30.0, "description": "High"}, {"value": 10.0, "description": "Low"},
    ]
    assert "gold-must-not-be-sent" not in str(questions)


def test_duplicate_text_preserves_each_candidate_position():
    case = cases()
    descriptions = sieve_questions(case)[0][1]
    assert len(descriptions) == len(case.questions[0].options)
    assert descriptions[0] == descriptions[1]


def test_overflow_rejected_before_native_prefix_or_trim():
    adapter = SieveAdapter.__new__(SieveAdapter)
    adapter.native = SimpleNamespace(choice_prefix=lambda state, question: state)
    adapter.engine = SimpleNamespace(list_options=False, _ids=lambda value: list(value), max_len=4)
    with pytest.raises(ValueError, match="5 tokens; refusing truncation"):
        adapter._probabilities("abc", "instruction", ["ab", "cd"])


def test_empty_tokenization_before_forward():
    adapter = SieveAdapter.__new__(SieveAdapter)
    adapter.native = SimpleNamespace(choice_prefix=lambda state, question: state)
    adapter.engine = SimpleNamespace(list_options=False, _ids=lambda value: list(value), max_len=4)
    with pytest.raises(ValueError, match="tokenization is empty"):
        adapter._probabilities("abc", "instruction", ["", "a"])


def test_explicit_dependency_revision_before_load():
    with pytest.raises(ValueError, match="explicit base revision"):
        SieveAdapter("unused", "main", "unused", "cuda:0", base_revision="")


def test_invalid_budget_before_load():
    with pytest.raises(ValueError, match="positive"):
        SieveAdapter("unused", "main", "unused", "cuda:0", base_revision="sha", context_limit=0)


def test_bounded_forks_use_independent_prefix_and_one_global_softmax():
    torch = pytest.importorskip("torch")
    adapter = SieveAdapter.__new__(SieveAdapter)
    adapter.torch = torch
    adapter.native = SimpleNamespace(choice_prefix=lambda state, question: state)
    seen = []

    def prefix_pass(prefixes):
        return {"fresh": True}, None

    def fork_pass(cache, attention, lengths, group, sources):
        assert cache.pop("fresh")
        seen.append(len(group))
        return torch.tensor([float(option[0]) for option in group])

    adapter.engine = SimpleNamespace(
        list_options=False, max_len=100, T=2.0,
        _ids=lambda value: [int(value)], _prefix_pass=prefix_pass, _fork_pass=fork_pass,
        head=lambda hidden: hidden,
    )
    probabilities = adapter._probabilities("0", "instruction", [str(i) for i in range(10)])
    assert seen == [8, 2]
    assert probabilities == pytest.approx(torch.softmax(torch.arange(10) / 2.0, dim=-1).tolist())
