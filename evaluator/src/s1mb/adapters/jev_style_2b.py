"""Pinned Jev-Style 2B block-causal attention and native calibrated verdicts."""

import importlib

from .jev_style import JevStyleAdapter


class JevStyle2BAdapter(JevStyleAdapter):
    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Jev-Style 2B context_limit must be positive")
        self.setup("jev-style-2b", model, revision, source, device)
        self.native = importlib.import_module("jev_style_decision")
        self.engine = self.native.JevStyleDecision(
            str(self.path), device=device, dtype="float32",
            max_len=context_limit or self.native.CONTEXT_LIMIT,
            verify=True, keep_state=True, cuda_graphs=False,
        )
        self.attention_model = self.engine.model
        if self.attention_model.config._attn_implementation != self.native.ATTN_NAME:
            raise RuntimeError("Jev-Style 2B native block-causal SDPA is not active")
        self.settings.update({
            "dtype": "float32", "attention_implementation": self.native.ATTN_NAME,
            "attention_policy": "native-block-causal-sdpa-no-block-resplitting",
            "block_size": self.native.BLOCK,
            "max_input_tokens": context_limit or self.native.CONTEXT_LIMIT,
            "input_length_policy": "reject-overflow-native",
            "option_chunk_policy": "native-complete-catalogue-and-rubric-blocks",
            "calibration_policy": "native-global-no-category",
            "temperature": self.engine.default_temperature,
            "cuda_graphs": False, "keep_state": True,
            "renderer": "native-structured-anonymous-numeric-score-v1",
        })

    def close(self):
        if hasattr(self, "engine"):
            self.engine.close()
        super().close()
