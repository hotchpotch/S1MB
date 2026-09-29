"""Zefan Cai's calibrated scalar head with bounded independent candidate batches."""

import importlib
import json

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches


class OpenJevAdapter(UpstreamAdapter):
    case_batch_size = 16

    def __init__(self, model, revision, source, device, context_limit=32768, case_batch_size=16):
        if case_batch_size < 1:
            raise ValueError("case_batch_size must be positive")
        self.case_batch_size = case_batch_size
        self.setup("open-jev", model, revision, source, device)
        for name in ("JEV_DEVICE_MAP", "JEV_LOAD_8BIT", "JEV_LOAD_4BIT"):
            import os

            if os.environ.get(name):
                raise ValueError(
                    f"Unset {name}; this evaluation requires unquantized single-GPU inference"
                )
        self.api = importlib.import_module("jev.api")
        checkpoint = self.path / "package/checkpoint"
        self.engine = importlib.import_module("jev.model").DecisionModel.load(checkpoint, device)
        self.attention_model = self.engine.backbone
        self.temperature = json.loads((checkpoint / "temperature.json").read_text())["temperature"]
        self.context_limit = context_limit
        self.settings = {
            "dtype": "bfloat16",
            "temperature": self.temperature,
            "checkpoint_max_length": self.engine.max_length,
            "max_length": context_limit,
            "input_length_policy": "reject-overflow",
            "prefix_cache": False,
            "case_batch_size": self.case_batch_size,
            "microbatch_tokens": 4096,
            "renderer": "native-independent-candidates-anonymous-choice-v1",
            "base_model": self.engine.model_id,
            "base_revision": self.engine.revision,
        }

        self.enable_kernels()

    def predict(self, case):
        return self.predict_batch([case])[0]

    def predict_batch(self, cases):
        if len(cases) > self.case_batch_size:
            return [
                output
                for offset in range(0, len(cases), self.case_batch_size)
                for output in self.predict_batch(cases[offset : offset + self.case_batch_size])
            ]
        texts, refs = [], []
        for index, case in enumerate(cases):
            for q in case.questions:
                definition = questions_for_api([q], structured=True)[q.id]
                if q.task == "choice":
                    definition["criteria"] = {
                        f"option_{i}": v for i, v in enumerate(definition["criteria"].values())
                    }
                record = self.api.compile_request(case.state, {"question": definition})[0]
                prompts = self.api.candidate_prompts(record)
                refs.append((index, case.case_id, q, len(prompts)))
                texts.extend(
                    self.engine.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for p in prompts
                )
        tokens = self.engine.tokenizer(texts, truncation=False)["input_ids"]
        if max(map(len, tokens)) > self.context_limit:
            raise ValueError("Open-Jev input exceeds context limit; refusing truncation")
        # Keep native row order: BF16 reordering can change calibrated probabilities.
        ordered = list(enumerate(tokens))
        scores = {}
        with self.torch.inference_mode():
            for batch in candidate_batches(ordered, [len(row[1]) for row in ordered], 16, 4096):
                encoded = self.engine.tokenizer.pad(
                    {"input_ids": [row[1] for row in batch]}, padding=True, return_tensors="pt"
                ).to(self.device)
                hidden = self.engine.backbone(**encoded, use_cache=False).last_hidden_state
                last = encoded["attention_mask"].sum(-1) - 1
                pooled = hidden[self.torch.arange(len(batch), device=self.device), last]
                values = self.engine.head(pooled.float()).flatten()
                for (i, _), value in zip(batch, values, strict=True):
                    scores[i] = value
            flat = self.torch.stack([scores[i] for i in range(len(tokens))])
            outputs = [[] for _ in cases]
            offset = 0
            for index, case_id, q, count in refs:
                logits = flat[offset : offset + count]
                offset += count
                if q.task == "noul":
                    logits = self.torch.stack([self.torch.zeros_like(logits[0]), logits[0]])
                probabilities = (logits.float() / self.temperature).softmax(-1).cpu().tolist()
                ids = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
                outputs[index].append(
                    Prediction(
                        case_id=case_id,
                        question_id=q.id,
                        probabilities=dict(zip(ids, probabilities, strict=True)),
                    )
                )
        return outputs
