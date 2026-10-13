"""Authored native questions, Beta bin geometry and no-truncation admission."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.kodiak import (
    KodiakAdapter,
    KodiakInputTooLong,
    beta_level_probabilities,
    kodiak_question,
)


def test_criteria_and_numeric_levels_remain_authored_and_anonymous():
    case = cases()
    choice = kodiak_question(case.questions[0])
    assert choice["labels"] == [{"id": "option_0", "text": "Same"},
                                {"id": "option_1", "text": "Same"}]
    assert choice["allow_null"] is False
    assert kodiak_question(case.questions[1])["labels"] == [
        {"id": "option_0", "text": "Yes"}, {"id": "option_1", "text": "No"},
    ]
    score = kodiak_question(case.questions[2])
    assert (score["min"], score["max"]) == (10.0, 30.0)
    assert (score["min_label"], score["max_label"]) == ("Low", "High")
    assert '"value": 10.0' in score["text"]
    assert "gold-must-not-be-sent" not in str([choice, score])


def test_beta_bins_use_numeric_midpoints_and_preserve_authored_ids():
    question = SimpleNamespace(options=[
        SimpleNamespace(id="high", value=40.0), SimpleNamespace(id="low", value=10.0),
        SimpleNamespace(id="middle", value=20.0),
    ])

    def uniform_cdf(value, alpha, beta):
        assert (alpha, beta) == (1.0, 1.0)
        return value

    probabilities = beta_level_probabilities(question, 0.5, 2.0, uniform_cdf)
    assert probabilities == pytest.approx({"low": 1 / 6, "middle": 1 / 2, "high": 1 / 3})


def test_invalid_beta_parameters_rejected():
    question = cases().questions[2]
    with pytest.raises(ValueError, match="invalid Beta"):
        beta_level_probabilities(question, 0.0, 2.0, lambda *args: 0)


def test_native_packer_truncation_rejected_before_gpu_forward():
    adapter = KodiakAdapter.__new__(KodiakAdapter)
    adapter.schema = cast(Any, SimpleNamespace(Request=SimpleNamespace(
        model_validate=lambda request: SimpleNamespace(model_dump=lambda: request),
    )))
    adapter.limits = None
    adapter.packing = cast(Any, SimpleNamespace(
        pack_example=lambda *args, **kw: SimpleNamespace(truncated=True),
    ))
    with pytest.raises(KodiakInputTooLong, match="refusing truncation"):
        adapter.predict(cases())


def test_explicit_character_budget_preserves_candidates_and_numeric_range():
    from s1mb.adapters.kodiak import budget_kodiak_characters

    native = {'type': 'score', 'id': 'decision', 'text': 'HEAD' + 'x' * 1000 + 'TAIL',
              'min': 10.0, 'max': 30.0, 'min_label': 'a' * 300, 'max_label': 'b' * 300}
    result, changes = budget_kodiak_characters(native, 500, 200)
    assert len(result['text']) == 500
    assert result['text'].startswith('HEAD') and result['text'].endswith('TAIL')
    assert (result['min'], result['max']) == (10.0, 30.0)
    assert len(result['min_label']) == len(result['max_label']) == 200
    assert len(changes) == 3
    assert len(native['text']) > 500
    choice = {'type': 'choice', 'text': 'q', 'labels': [{'id': 'a', 'text': 'x' * 300}, {'id': 'b', 'text': 'y' * 300}]}
    result, _ = budget_kodiak_characters(choice, 500, 200)
    assert [item['id'] for item in result['labels']] == ['a', 'b']
    assert all(len(item['text']) == 200 for item in result['labels'])


def test_character_budget_distinguishes_collapsed_label_descriptions():
    from s1mb.adapters.kodiak import budget_kodiak_characters

    native = {'type': 'choice', 'text': 'q', 'labels': [
        {'id': 'a', 'text': 'x' * 150 + 'first' + 'y' * 150},
        {'id': 'b', 'text': 'x' * 150 + 'second' + 'y' * 150},
    ]}
    result, _ = budget_kodiak_characters(native, 500, 200)
    assert [label['id'] for label in result['labels']] == ['a', 'b']
    assert len({label['text'].strip().casefold() for label in result['labels']}) == 2
    assert all(len(label['text']) <= 200 for label in result['labels'])
