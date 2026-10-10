"""Needle parameter counts exclude all embeddings even when shared with output heads."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.needle import NeedleAdapter, needle_parameter_metadata
from s1mb.parameters import METHOD


def tensor(*shape):
    return SimpleNamespace(shape=shape)


def test_token_and_engram_embeddings_are_excluded():
    params = {
        "embedding": {"embedding": tensor(10, 4)},
        "engrams_0": {"embedding": tensor(3, 5), "k_proj": {"kernel": tensor(5, 4)}},
        "engrams_1": {"embedding": tensor(2, 5)},
        "stack": {"weight": tensor(4, 4)},
        "confidence_head": {"bias": tensor()},
    }
    assert needle_parameter_metadata(params) == {
        "total_params": 102,
        "active_params": 37,
        "parameter_count_method": METHOD,
    }


def test_shared_embedding_is_counted_once_and_excluded():
    shared = tensor(3, 4)
    params = {"engrams_0": {"embedding": shared}, "output": {"kernel": shared}}
    assert needle_parameter_metadata(params)["total_params"] == 12
    assert needle_parameter_metadata(params)["active_params"] == 0


def test_empty_parameters_are_rejected():
    with pytest.raises(ValueError, match="No Needle parameters"):
        needle_parameter_metadata({})


def test_adapter_metadata_includes_counts_without_inference():
    adapter = NeedleAdapter.__new__(NeedleAdapter)
    adapter.model_id, adapter.revision, adapter.settings = "test/needle", "revision", {}
    adapter.parameter_counts = needle_parameter_metadata({"embedding": {"embedding": tensor(2, 3)}})
    info = adapter.metadata()
    assert info.total_params == 6 and info.active_params == 0
    assert info.parameter_count_method == METHOD
