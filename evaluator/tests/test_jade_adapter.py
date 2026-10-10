"""JADE must combine native candidate chunks without losing or estimating labels."""

import pytest

from s1mb.adapters.jade import jade_candidate_logits


def test_chunked_readout_retains_order_and_rejects_missing_candidate():
    chunks = [([8, 2], {"token_id:8": -0.2, "token_id:2": -1.2}),
              ([4], {"token_id:4": -2.2})]
    assert jade_candidate_logits(chunks, [4, 8, 2]) == [-2.2, -0.2, -1.2]
    with pytest.raises(ValueError, match="omitted a requested candidate"):
        jade_candidate_logits([([8, 2], {"token_id:8": -0.2})], [8, 2])
    with pytest.raises(ValueError, match="duplicate or nonfinite"):
        jade_candidate_logits([([8], {"token_id:8": -0.2}), ([8], {"token_id:8": -0.2})], [8])
