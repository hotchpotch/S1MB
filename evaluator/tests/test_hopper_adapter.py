"""Hopper refuses invented label capacity before loading native weights."""

import pytest

from s1mb.adapters.hopper import HopperAdapter


def test_native_label_capacity_cannot_be_extended():
    with pytest.raises(ValueError, match="26 native"):
        HopperAdapter("model", "revision", "source", "cuda:0", max_candidates=27)
