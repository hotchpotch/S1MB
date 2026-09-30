"""Check that query tiling preserves relative attention and padding semantics."""

import copy

import pytest


@pytest.mark.parametrize("length", [19, 71])
def test_tiled_relative_attention_matches_native(length):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from s1mb.adapters.deberta_attention import enable_query_tiling

    torch.manual_seed(3)
    config = transformers.DebertaV2Config(
        vocab_size=48,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=32,
        position_buckets=8,
        relative_attention=True,
        position_biased_input=False,
        share_att_key=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    reference = transformers.DebertaV2Model(config).eval()
    tiled = copy.deepcopy(reference)
    enable_query_tiling(tiled, block_size=7)
    ids = torch.randint(1, config.vocab_size, (2, length))
    mask = torch.ones_like(ids)
    mask[1, -4:] = 0
    with torch.inference_mode():
        expected = reference(ids, attention_mask=mask).last_hidden_state
        actual = tiled(ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
