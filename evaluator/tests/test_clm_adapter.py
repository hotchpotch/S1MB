"""Public CPU checks for the CLM pooling and bounded input boundary."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.clm import CLMAdapter, last_token_embeddings


def test_last_token_pooling_ignores_padding_on_either_side():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor(
        [[[3.0, 4.0], [0.0, 5.0], [99.0, 99.0]], [[99.0, 99.0], [0.0, 5.0], [3.0, 4.0]]]
    )
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    actual = last_token_embeddings(hidden, mask)
    torch.testing.assert_close(actual, torch.tensor([[0.0, 1.0], [0.6, 0.8]]))


def test_last_token_pooling_rejects_empty_sequence():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="empty token sequence"):
        last_token_embeddings(torch.zeros(1, 3, 2), torch.zeros(1, 3, dtype=torch.long))


def test_overflow_rejected_before_model_call():
    adapter = CLMAdapter.__new__(CLMAdapter)
    adapter.context_limit = 4

    def tokenize(texts, truncation):
        assert truncation is False
        return {"input_ids": [[1] * len(text) for text in texts]}

    adapter.tokenizer = tokenize
    adapter.engine = SimpleNamespace()
    with pytest.raises(ValueError, match="token length"):
        adapter.embed(["five!"])
