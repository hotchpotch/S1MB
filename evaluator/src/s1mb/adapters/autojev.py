"""AutoJev's published native decision backbone, readout and temperature."""

import importlib
import json
import sys

from .firelex_jeff import FirelexJeffAdapter


class AutoJevAdapter(FirelexJeffAdapter):
    def __init__(self, model, revision, source, device, context_limit=None):
        if sys.version_info < (3, 12):
            raise RuntimeError("AutoJev's upstream runtime requires Python 3.12 or later")
        self.setup("autojev", model, revision, source, device)
        config = json.loads((self.path / "decision_config.json").read_text())
        self.native = importlib.import_module("autojev.model")
        self.engine = self.native.DecisionModel(checkpoint=self.path, device=device).eval()
        self.attention_model = self.engine.backbone
        self.max_candidates = min(self.native.MAX_OPTIONS, len(self.engine.codes))
        self.context_limit = context_limit or 8192
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "temperature": self.engine.temperature,
            "max_input_tokens": self.context_limit,
            "max_candidates": self.max_candidates,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "base_model": config["base_model"],
            "base_revision": config["revision"],
            "renderer": "native-anonymous-choice-numeric-score-v1",
        }
        self.enable_kernels()
