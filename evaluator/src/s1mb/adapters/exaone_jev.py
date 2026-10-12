"""EXAONE-JEV native calibrated permutation and multi-round label scoring."""

import asyncio
import importlib
import importlib.util
import json
import math
from typing import Any, cast

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


def exaone_questions(case):
    questions = questions_for_api(case.questions, structured=True, anonymous_choice=True)
    for question in case.questions:
        if question.task == "score":
            item = questions[question.id]
            item["criteria"] = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, item["criteria"], strict=True)
            ]
    return questions


class ExaoneJevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("EXAONE-JEV context_limit must be positive")
        self.setup("exaone-jev", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.path)
        if self.tokenizer is None:
            raise ValueError("EXAONE-JEV tokenizer is unavailable")
        self.engine = transformers.AutoModelForCausalLM.from_pretrained(
            self.path, dtype=self.torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
        self.attention_model = self.engine
        self.context_limit = context_limit or 32768
        if self.context_limit > self.engine.config.max_position_embeddings:
            raise ValueError("EXAONE-JEV context limit exceeds checkpoint capacity")
        # A private module keeps its native S/read globals isolated per adapter.
        spec = importlib.util.spec_from_file_location(
            "_s1mb_exaone_native", self.source / "serve/systemone_server.py",
        )
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load EXAONE-JEV native runtime")
        self.native = cast(Any, importlib.util.module_from_spec(spec))
        spec.loader.exec_module(self.native)
        default, by_type = self.native.load_temperature(str(self.path), None)
        if any(not math.isfinite(value) or value <= 0 for value in [default, *by_type.values()]):
            raise ValueError("Invalid native EXAONE-JEV calibration")
        config = json.loads((self.path / "jev_config.json").read_text())
        permute = config.get("permute", 1)
        if permute not in {1, 2}:
            raise ValueError("Unsupported native EXAONE-JEV permutation count")
        labels = [self.tokenizer.encode(label, add_special_tokens=False)
                  for label in self.native.D.LETTERS[:self.native.D.MAX_LABELS]]
        if any(len(label) != 1 for label in labels) or len({label[0] for label in labels}) != len(labels):
            raise ValueError("EXAONE-JEV labels must be distinct single tokens")
        self.native.S.update({
            "tok": self.tokenizer, "label_ids": [label[0] for label in labels],
            "T": default, "T_by_type": by_type, "max_len": self.context_limit,
            "permute": permute,
        })
        self.native.read = self._read
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit, "input_length_policy": "reject-overflow-native",
            "temperature": default, "temperature_by_type": by_type, "permute": permute,
            "runtime_backend": "transformers-native-label-logprobs",
            "readout": "native-calibrated-original-reverse-balanced-chunks-and-final-pool",
            "max_candidates": self.native.D.MAX_OPTIONS,
            "max_labels_per_read": self.native.D.MAX_LABELS,
            "renderer": "native-tev1-structured-anonymous-numeric-score",
        })

    async def _read(self, prompt_ids, n_labels):
        if len(prompt_ids) + 1 > self.context_limit:
            raise ValueError("EXAONE-JEV input overflow; refusing truncation")
        batch = self.torch.tensor([prompt_ids], dtype=self.torch.long, device=self.device)
        with self.torch.inference_mode():
            logits = self.engine(batch, use_cache=False).logits[0, -1].float()
            logprobs = self.torch.log_softmax(logits, dim=-1)
            return logprobs[self.native.S["label_ids"][:n_labels]].cpu().tolist()

    def predict(self, case):
        answers = {}
        for key, question in exaone_questions(case).items():
            answers[key], _ = asyncio.run(self.native.ask(case.state, question))
        return decode_answers(case, answers, anonymous_choice=True)

    def close(self):
        if hasattr(self, "native"):
            # Break the callback cycle before releasing GPU modules.
            self.native.read = None
            self.native.S.clear()
        super().close()
