"""Selected-token probabilities must be complete and aligned."""

import pytest

from s1mb.adapters.eikos_fp8 import eikos_fp8_probabilities


def test_omitted_candidate_is_rejected():
    with pytest.raises(ValueError, match="omitted requested candidate"):
        eikos_fp8_probabilities([{"token": "token_id:1", "logprob": -0.1}], [1, 2], 1.3)


def test_selected_logprobs_preserve_order_and_calibration():
    rows = [{"token": "token_id:9", "logprob": -1.3},
            {"token": "token_id:4", "logprob": -2.6},
            {"token": "token_id:2", "logprob": -0.1}]
    assert eikos_fp8_probabilities(rows, [4, 9], 1.3) == pytest.approx(
        [0.2689414213699951, 0.7310585786300049],
    )
