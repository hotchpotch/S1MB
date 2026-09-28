"""Explicit branch limits must never silently remove query/candidate tokens."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.input_lengths import check_input_lengths


def test_balanced_budget_includes_markers_and_both_fields():
    from s1mb.adapters.input_lengths import check_balanced_input_lengths_batched

    parts = [SimpleNamespace(system="Policy", instruction="Judge", context="a b c")]
    # Two markers, one system token, four field tokens and two special tokens.
    check_balanced_input_lengths_batched(tokenizer, parts, 9)
    with pytest.raises(ValueError, match="Balanced query requires 9"):
        check_balanced_input_lengths_batched(tokenizer, parts, 8)


def tokenizer(texts, *, add_special_tokens, truncation):
    assert not add_special_tokens and not truncation
    return {"input_ids": [list(range(len(text.split()))) for text in texts]}


def test_limits_include_special_tokens_and_accept_exact_fit():
    check_input_lengths(tokenizer, ["a b c", "a b c"], ["x y"], 5, 3)


@pytest.mark.parametrize(
    ("queries", "documents", "message"),
    [(["a b c d"], ["x"], "query requires 6"), (["a"], ["x y z"], "candidate requires 4")],
)
def test_overflow_is_rejected(queries, documents, message):
    with pytest.raises(ValueError, match=message):
        check_input_lengths(tokenizer, queries, documents, 5, 3)


def test_batched_balanced_validation_preserves_duplicate_field_lengths():
    from s1mb.adapters.input_lengths import check_balanced_input_lengths_batched

    parts = [SimpleNamespace(system="", instruction="same words", context="same words")] * 3
    calls = []

    def recording(texts, **kwargs):
        calls.append(texts)
        return tokenizer(texts, **kwargs)

    check_balanced_input_lengths_batched(recording, parts, 8)
    assert len(calls) == 1 and len(calls[0]) == len(set(calls[0]))
    with pytest.raises(ValueError, match="Balanced query requires 8"):
        check_balanced_input_lengths_batched(recording, parts, 7)
    check_balanced_input_lengths_batched(recording, [], 2)
    assert len(calls) == 2
