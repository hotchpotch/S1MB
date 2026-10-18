"""Reject offload and retain activation space in native decoder placement."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.cuda_sharding import dispatch_decision_tower, two_gpu_cut


def test_contiguous_split_accounts_for_vision_and_embeddings():
    assert two_gpu_cut([10] * 6, 20, [60, 60]) == 2


def test_rejects_weights_that_only_fit_without_activation_reserve():
    with pytest.raises(ValueError, match="CPU offload is disabled"):
        two_gpu_cut([30, 30], 0, [33, 33])


def test_requires_two_devices():
    with pytest.raises(ValueError, match="Two CUDA"):
        two_gpu_cut([10, 10], 0, [100])


def test_rejects_unknown_native_decision_tower_before_dispatch():
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
    with pytest.raises(ValueError, match="decision tower structure"):
        dispatch_decision_tower(SimpleNamespace(), torch)
