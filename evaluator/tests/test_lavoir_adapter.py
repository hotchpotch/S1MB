"""Preserve authored typed definitions and reject overflow before model calls."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.lavoir import LavoirAdapter, lavoir_question


def test_numeric_score_structured_choice_and_authored_noul():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    case.questions[2].options.reverse()
    items = [lavoir_question(q) for q in case.questions]
    assert items[0]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert items[1]["criteria"] == {"true": "Yes", "false": "No"}
    assert items[2]["criteria"] == ["30.0: High", "10.0: Low"]
    assert "gold-must-not-be-sent" not in str(items)


@pytest.mark.parametrize("limit", [0, 1025])
def test_invalid_budget_before_download(limit):
    with pytest.raises(ValueError):
        LavoirAdapter("unused", "main", "unused", "cuda:0", context_limit=limit)


def test_overflow_never_reaches_collation_or_inference(monkeypatch):
    model = LavoirAdapter.__new__(LavoirAdapter)
    model.engine = SimpleNamespace(tokenizer=None)
    monkeypatch.setattr(
        model, "items", SimpleNamespace(to_internal=lambda item: item), raising=False
    )
    monkeypatch.setattr(model, "common", None, raising=False)
    model.context_limit = 1024

    def overflowing(*args):
        raise ValueError("Laya input requires 1025 tokens, exceeding 1024")

    monkeypatch.setattr("s1mb.adapters.lavoir.full_sequence", overflowing)
    with pytest.raises(ValueError, match="exceeding 1024"):
        model.predict(cases())
