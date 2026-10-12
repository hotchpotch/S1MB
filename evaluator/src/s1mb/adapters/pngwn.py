"""Pngwn's scalar option scorer with full-input split tokenization."""

import importlib
import json
import math
from typing import Any, cast

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, candidate_batches, checkpoint_path, state_text


class PngwnAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, base_revision=None):
        if not base_revision:
            raise ValueError("Pngwn requires an explicit base checkpoint revision")
        self.setup("pngwn", model, revision, source, device)
        transformers = cast(Any, importlib.import_module("transformers"))
        peft = importlib.import_module("peft")
        base = json.loads((self.path / "adapter_config.json").read_text())[
            "base_model_name_or_path"
        ]
        base_path, base_revision = checkpoint_path(base, base_revision)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(base_path))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        backbone = transformers.Qwen3_5TextForSequenceClassification.from_pretrained(
            str(base_path),
            num_labels=1,
            dtype=self.torch.bfloat16,
            attn_implementation="sdpa",
        )
        backbone.config.pad_token_id = self.tokenizer.pad_token_id
        backbone.config.eos_token_id = self.tokenizer.eos_token_id
        text_config = backbone.config.get_text_config()
        text_config.pad_token_id = self.tokenizer.pad_token_id
        text_config.eos_token_id = self.tokenizer.eos_token_id
        text_config.num_labels = 1
        backbone.num_labels = 1
        self.engine = peft.PeftModel.from_pretrained(backbone, str(self.path)).to(device).eval()
        self.attention_model = self.engine
        self.temperature = json.loads((self.path / "metrics.json").read_text())["temperature"]
        if not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Pngwn release temperature must be finite and positive")
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": 32768,
            "case_batch_size": 1,
            "temperature": self.temperature,
            "microbatch_tokens": 4096,
            "base_model": base,
            "base_revision": base_revision,
            "renderer": "native-split-state-question-option-v1",
            "extended_input_condition": "full fields to 32768 tokens; no demo field slicing",
        }
        self.enable_kernels()

    def predict(self, case):
        tok = self.tokenizer
        prefix = tok("State:\n" + state_text(case.state), add_special_tokens=False)["input_ids"]
        predictions = []
        for q in case.questions:
            instruction = "\n\n".join(
                x for x in (q.system_prompt, q.instructions_json or q.instructions) if x
            )
            rows = []
            for o in q.options:
                option = o.description_json or o.description
                if q.task == "score":
                    option = f"{o.value}: {option}"
                elif q.task == "noul":
                    option = f"{o.id}: {option}"
                suffix = "\n\nQuestion:\n" + instruction + "\n\nOption:\n" + option
                ids = prefix + tok(suffix, add_special_tokens=False)["input_ids"]
                if len(ids) > self.settings["max_input_tokens"]:
                    raise ValueError("Pngwn input exceeds context limit; refusing truncation")
                rows.append(ids)
            logits = []
            with self.torch.inference_mode():
                for group in candidate_batches(rows, list(map(len, rows)), 16, 4096):
                    width = max(map(len, group))
                    ids = self.torch.tensor(
                        [r + [tok.pad_token_id] * (width - len(r)) for r in group],
                        device=self.device,
                    )
                    mask = self.torch.tensor(
                        [[1] * len(r) + [0] * (width - len(r)) for r in group], device=self.device
                    )
                    logits.extend(
                        self.engine(input_ids=ids, attention_mask=mask, use_cache=False)
                        .logits[:, 0]
                        .float()
                        .cpu()
                        .tolist()
                    )
            raw = (self.torch.tensor(logits, dtype=self.torch.float64) / self.temperature).softmax(-1).tolist()
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: p for o, p in zip(q.options, raw, strict=True)},
                )
            )
        return predictions
