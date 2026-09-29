"""Minojev's general checkpoint and typed candidate distributions."""

import importlib
from dataclasses import replace
from typing import Any

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches


def fp32_sdpa(module, query, key, value, attention_mask, **kwargs):
    """Expand GQA explicitly so FP32 SDPA can use its memory-efficient kernel."""
    torch = importlib.import_module("torch")
    attention = importlib.import_module("torch.nn.attention")
    sdpa = importlib.import_module("transformers.integrations.sdpa_attention")

    groups = query.shape[1] // key.shape[1]
    key, value = sdpa.repeat_kv(key, groups), sdpa.repeat_kv(value, groups)
    if attention_mask is not None and attention_mask.stride(-2) % 8:
        width = attention_mask.shape[-1]
        attention_mask = torch.nn.functional.pad(attention_mask, (0, -width % 8))[..., :width]
    proxy: Any = torch.nn.Module()
    proxy.is_causal = getattr(module, "is_causal", True)
    with attention.sdpa_kernel(attention.SDPBackend.EFFICIENT_ATTENTION):
        return sdpa.sdpa_attention_forward(proxy, query, key, value, attention_mask, **kwargs)


class MinojevAdapter(UpstreamAdapter):
    case_batch_size = 16

    def __init__(self, model, revision, source, device, dtype="float32"):
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("Minojev dtype must be float32 or bfloat16")
        self.setup("minojev", model, revision, source, device)
        module = importlib.import_module("minojev.model")
        self.types = importlib.import_module("minojev.types")
        path = self.path / "general" if (self.path / "general/config.json").exists() else self.path
        self.engine = module.DecisionModel.load(path, device=device)
        self.attention_model = self.engine.backbone.model
        if dtype == "bfloat16":
            # Keep the trained decision head in FP32, as required by its native forward path.
            self.engine.backbone.model.to(dtype=self.torch.bfloat16)
        self.options = module.ScoreOptions(mode="fresh", batch_requests=128, device=device)
        native_forward = self.engine.forward_paths
        cache_type = importlib.import_module("transformers").DynamicCache
        pad_sequences = importlib.import_module("minojev.encoding").pad_sequences

        def bounded_forward(encoded):
            # Bound backbone memory while retaining the head's joint candidate attention.
            if max(len(path.token_ids) for path in encoded.paths) > 32768:
                raise ValueError("Minojev input exceeds context limit; refusing truncation")
            hidden = {}
            prefixes = {}
            for index, path in enumerate(encoded.paths):
                prefixes.setdefault(tuple(path.state_ids), []).append((index, path))
            fresh = []
            for prefix, paths in prefixes.items():
                if len(prefix) < 512 or len(paths) < 4:
                    fresh.extend(paths)
                    continue
                tokens = self.torch.tensor([prefix], device=device)
                _, prefix_cache = self.engine.backbone(input_ids=tokens, use_cache=True)
                ordered = sorted(paths, key=lambda row: len(row[1].suffix_ids))
                for batch in candidate_batches(
                    ordered, [len(row[1].token_ids) for row in ordered], 16, 8192
                ):
                    cache = cache_type()
                    for layer_index, layer in enumerate(prefix_cache.layers):
                        cache.update(
                            layer.keys.expand(len(batch), -1, -1, -1).contiguous(),
                            layer.values.expand(len(batch), -1, -1, -1).contiguous(),
                            layer_index,
                        )
                    suffix, mask = pad_sequences(
                        [row[1].suffix_ids for row in batch], self.engine.tokenizer.pad_id, device
                    )
                    full_mask = self.torch.cat(
                        [mask.new_ones((len(batch), len(prefix))), mask], dim=1
                    )
                    values, _ = self.engine.backbone(
                        input_ids=suffix,
                        attention_mask=full_mask,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    rows = values[
                        self.torch.arange(len(batch), device=device), mask.sum(-1) - 1
                    ].float()
                    for (index, _), value in zip(batch, rows, strict=True):
                        hidden[index] = value
                    del cache, values
                del prefix_cache
            ordered = sorted(fresh, key=lambda row: len(row[1].token_ids))
            for batch in candidate_batches(
                ordered, [len(row[1].token_ids) for row in ordered], 16, 8192
            ):
                values = native_forward(replace(encoded, paths=[row[1] for row in batch]))
                for (index, _), value in zip(batch, values, strict=True):
                    hidden[index] = value
            return self.torch.stack([hidden[i] for i in range(len(encoded.paths))])

        self.engine.forward_paths = bounded_forward
        self.settings = {
            "config": self.engine.config,
            "dtype": str(next(self.engine.backbone.parameters()).dtype),
            "calibration": self.engine.calibration.to_dict(),
            "mode": "shared-prefix-for-long-inputs",
            "prefix_cache_min_tokens": 512,
            "prefix_cache_min_paths": 4,
            "candidate_batch_size": 16,
            "candidate_batch_token_budget": 8192,
            "case_batch_size": self.case_batch_size,
            "max_input_tokens": 32768,
            "input_length_policy": "reject-overflow",
            "checkpoint_subdir": str(path.relative_to(self.path)),
        }

    def predict(self, case):
        return self.predict_batch([case])[0]

    def set_attention(self, implementation):
        if implementation == "sdpa" and self.settings["dtype"] == "torch.float32":
            transformers = importlib.import_module("transformers")
            masking = importlib.import_module("transformers.masking_utils")
            transformers.AttentionInterface.register("s1mb_fp32_sdpa", fp32_sdpa)
            masking.AttentionMaskInterface.register(
                "s1mb_fp32_sdpa", masking.AttentionMaskInterface()["sdpa"]
            )
            implementation = "s1mb_fp32_sdpa"
        super().set_attention(implementation)

    def predict_batch(self, cases):
        requests, refs = [], []
        for index, case in enumerate(cases):
            for q in case.questions:
                native = questions_for_api([q])
                if q.task == "score":
                    native[q.id]["levels"] = native[q.id].pop("criteria")
                requests.append(
                    self.types.request_from_object(
                        {"id": case.case_id, "state": case.state, "questions": native}
                    )
                )
                refs.append((index, case.case_id, q))
        outputs = [[] for _ in cases]
        for (index, case_id, q), row in zip(
            refs, self.engine.score(requests, self.options), strict=True
        ):
            ids = row["candidate_ids"] if q.task != "score" else [o.id for o in q.options]
            p = dict(zip(ids, row["probabilities"], strict=True))
            # Account for native FP32 softmax error as well as eight-decimal rounding.
            if abs(sum(p.values()) - 1) > len(p) * 0.5e-8 + 1e-6:
                raise ValueError("Minojev distribution exceeds rounding tolerance")
            total = sum(p.values())
            p = {k: v / total for k, v in p.items()}
            check_probabilities(p, [o.id for o in q.options])
            outputs[index].append(Prediction(case_id=case_id, question_id=q.id, probabilities=p))
        return outputs
