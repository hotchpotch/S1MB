"""Memory-bounded, exact query tiling for inference with DeBERTa relative attention."""

import importlib
import math
from types import MethodType


def enable_query_tiling(backbone, block_size=128):
    """Retain every key and relative-position term; split only the query dimension."""
    torch = importlib.import_module("torch")
    native = importlib.import_module("transformers.models.deberta_v2.modeling_deberta_v2")
    if backbone.config.position_biased_input or not backbone.config.relative_attention:
        raise ValueError("Query tiling requires relative-only DeBERTa attention")
    if block_size < 1:
        raise ValueError("Query block size must be positive")

    def forward(
        module,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        if (
            module.training
            or output_attentions
            or query_states is not None
            or relative_pos is not None
        ):
            raise ValueError(
                "Tiled DeBERTa supports inference self-attention without attention output"
            )
        if rel_embeddings is None:
            raise ValueError("Tiled DeBERTa requires its learned relative embeddings")
        heads = module.num_attention_heads
        q = module.transpose_for_scores(module.query_proj(hidden_states), heads)
        k = module.transpose_for_scores(module.key_proj(hidden_states), heads)
        v = module.transpose_for_scores(module.value_proj(hidden_states), heads)
        factor = 1 + int("c2p" in module.pos_att_type) + int("p2c" in module.pos_att_type)
        scale = math.sqrt(q.shape[-1] * factor)
        span = module.pos_ebd_size
        embeddings = module.pos_dropout(rel_embeddings[: 2 * span]).unsqueeze(0)
        batch = hidden_states.shape[0]
        pos_k = pos_q = None
        if "c2p" in module.pos_att_type:
            projection = module.key_proj if module.share_att_key else module.pos_key_proj
            pos_k = module.transpose_for_scores(projection(embeddings), heads).repeat(batch, 1, 1)
        if "p2c" in module.pos_att_type:
            projection = module.query_proj if module.share_att_key else module.pos_query_proj
            pos_q = module.transpose_for_scores(projection(embeddings), heads).repeat(batch, 1, 1)
        key_position_scores = torch.bmm(k, pos_q.transpose(-1, -2)) if pos_q is not None else None
        length = hidden_states.shape[1]
        key_positions = torch.arange(length, device=q.device)
        chunks = []
        for start in range(0, length, block_size):
            stop = min(length, start + block_size)
            query = q[:, start:stop]
            relative = torch.arange(start, stop, device=q.device)[:, None] - key_positions[None, :]
            if module.position_buckets > 0 and module.max_relative_positions > 0:
                relative = native.make_log_bucket_position(
                    relative, module.position_buckets, module.max_relative_positions
                ).long()
            indices = (relative + span).clamp(0, 2 * span - 1)
            scores = torch.bmm(query, k.transpose(-1, -2) / scale)
            if pos_k is not None:
                relative_scores = torch.bmm(query, pos_k.transpose(-1, -2))
                scores += (
                    relative_scores.gather(2, indices.unsqueeze(0).expand(q.shape[0], -1, -1))
                    / scale
                )
            if key_position_scores is not None:
                # Native p2c gathers in key/query order, then transposes. For a
                # symmetric bucket function this uses bucket(query - key), too.
                relative_scores = key_position_scores.gather(
                    2, indices.T.unsqueeze(0).expand(q.shape[0], -1, -1)
                ).transpose(1, 2)
                scores += relative_scores / scale
            scores = scores.view(batch, heads, stop - start, length)
            mask = (
                attention_mask[..., start:stop, :]
                if attention_mask.shape[-2] != 1
                else attention_mask
            )
            scores = scores.masked_fill(~mask.bool(), torch.finfo(q.dtype).min)
            probabilities = scores.softmax(-1).reshape(batch * heads, stop - start, length)
            chunks.append(torch.bmm(probabilities, v))
        context = torch.cat(chunks, dim=1).view(batch, heads, length, q.shape[-1])
        return context.permute(0, 2, 1, 3).contiguous().view(batch, length, -1), None

    # Avoid materializing the full quadratic relative-position matrix upstream.
    backbone.encoder.get_rel_pos = lambda *args, **kwargs: None
    for layer in backbone.encoder.layer:
        layer.attention.self.forward = MethodType(forward, layer.attention.self)
