"""Eval Engine's released LoRA using its Tev-compatible decision protocol."""

import importlib
import json

from .tev import LABELS, TevAdapter
from .upstream import checkpoint_path


class EvalEngineAdapter(TevAdapter):
    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("evalengine", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        config = json.loads((self.path / "adapter_config.json").read_text())
        base = config["base_model_name_or_path"]
        base_path, base_revision = checkpoint_path(base, config.get("revision") or "main")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(base_path))
        if self.tokenizer is None:
            raise ValueError("Eval Engine base checkpoint did not provide a tokenizer")
        backbone = transformers.AutoModelForCausalLM.from_pretrained(
            str(base_path),
            dtype=self.torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        self.engine = peft.PeftModel.from_pretrained(backbone, str(self.path)).eval()
        self.attention_model = self.engine
        self.context_limit = context_limit or 2048
        encoded = [self.tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
        if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != len(LABELS):
            raise ValueError("Eval Engine requires distinct single-token answer letters")
        self.label_ids = [ids[0] for ids in encoded]
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "base_model": base,
            "base_revision": base_revision,
            "max_input_tokens": self.context_limit,
            "native_max_input_tokens": 2048,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "native-tev-json-anonymous-choice-v1",
            "temperature": 1.0,
            "logits_to_keep": 1,
            "large_menu_policy": "balanced-ordered-24-way-groups-product-of-conditionals-v1",
        }
        self.enable_kernels()
