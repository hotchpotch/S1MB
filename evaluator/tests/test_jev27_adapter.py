"""Native JEV 27B bias/calibration and complete candidate readout boundaries."""

import math

import pytest

from s1mb.adapters.jev27 import jev27_probabilities


def test_bias_and_temperature_apply_once_in_candidate_order():
    values = jev27_probabilities({"token_id:2": -3, "token_id:1": -4}, [1, 2], [3, 0], 2)
    expected = 1 / (1 + math.exp(-1))
    assert values == pytest.approx([expected, 1 - expected])


def test_missing_wide_candidate_is_rejected():
    with pytest.raises(ValueError, match="omitted"):
        jev27_probabilities({"token_id:1": -1}, [1, 2], [0, 0], 1)
