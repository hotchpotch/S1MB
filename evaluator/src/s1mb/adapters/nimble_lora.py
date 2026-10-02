"""Frozen Nimble schema scoring for independently published LoRA checkpoints."""

import importlib
import json
import string

from .jevlite import extended_labels
from .metask import MetaskAdapter
from .upstream import checkpoint_path


class NimbleLoraAdapter(MetaskAdapter):
    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("nimble-lora", model, revision, source, device)
        self.native = importlib.import_module("nimble.scoring.parallel_schema")
        loader = importlib.import_module("nimble.training.model_loading")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        contract = json.loads((self.path / "schema_config.json").read_text())
        base_path, base_revision = checkpoint_path(contract["model"], contract["revision"])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        base = loader.load_base(str(base_path), base_revision)
        self.engine = peft.PeftModel.from_pretrained(base, str(self.path)).eval()
        self.attention_model = self.engine
        self.context_limit = context_limit or contract["max_length"]
        self.codes = extended_labels(
            self.tokenizer, list(string.ascii_uppercase), max_candidates or 26
        )
        self.temperature = dict.fromkeys(["choice", "noul", "score"], 1.0)
        self.settings = {
            "dtype": "bfloat16",
            "max_input_tokens": self.context_limit,
            "checkpoint_runtime_limit": contract["max_length"],
            "base_model": contract["model"],
            "base_revision": base_revision,
            "max_candidates": len(self.codes),
            "native_max_candidates": 26,
            "answer_codes": self.codes,
            "temperature": self.temperature,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "native-nimble-enum-authored-noul-numeric-score-v1",
        }
        self.set_attention("sdpa")
        self.enable_kernels()
