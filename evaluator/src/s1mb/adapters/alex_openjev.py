"""Alex Wortega's typed NLI decisions with native windows and untruncated pairs."""

import importlib
import json
from typing import Any

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, candidate_batches, state_text


class AlexOpenJevAdapter(UpstreamAdapter):
    case_batch_size = 16

    def __init__(self, model, revision, source, device, context_limit=32768, subfolder=None):
        self.setup("alex-openjev", model, revision, source, device, subfolder)
        transformers = importlib.import_module("transformers")
        self.tokenizer: Any = transformers.AutoTokenizer.from_pretrained(self.path)
        assert self.tokenizer is not None
        self.tokenizer.padding_side = "right"
        self.engine = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                self.path,
                dtype=self.torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        self.attention_model = self.engine
        self.model_id = f"AlexWortega/openjev/{self.path.name}"
        self.context_limit = context_limit
        self.template = self.engine.config.nli_template
        self.entailment = self.engine.config.label2id["entailment"]
        self.settings = {
            "dtype": "bfloat16",
            "max_length": context_limit,
            "input_length_policy": "reject-overflow",
            "case_batch_size": self.case_batch_size,
            "microbatch_tokens": 4096,
            "window_chars": 24000,
            "window_overlap_chars": 2000,
            "renderer": "native-typed-nli-anonymous-choice-v1",
            "probability_rule": "max-window-entailment-normalized-over-options",
            "nli_template": self.template,
        }

        self.enable_kernels()

    def predict(self, case):
        return self.predict_batch([case])[0]

    def predict_batch(self, cases):
        texts, refs = [], []
        for index, case in enumerate(cases):
            state = state_text(case.state)
            windows = (
                [state]
                if len(state) <= 24000
                else [state[s : s + 24000] for s in range(0, max(len(state) - 2000, 1), 22000)]
            )
            for q in case.questions:
                instruction = "\n\n".join(s for s in [q.system_prompt, q.instructions] if s).strip()
                options = q.options
                labels = (
                    ["yes" if o.id == "true" else "no" for o in options]
                    if q.task == "noul"
                    else [f"option_{i}" for i in range(len(options))]
                )
                refs.append((index, case.case_id, q, len(windows)))
                for window in windows:
                    for label, option in zip(labels, options, strict=True):
                        description = option.description
                        if option.description_json:
                            value = json.loads(option.description_json)
                            description = (
                                value
                                if isinstance(value, str)
                                else json.dumps(value, ensure_ascii=False)
                            )
                        hypothesis = f'The answer to "{instruction}" is {label}: {description}'
                        texts.append(
                            self.template.format(
                                premise=window.strip(), hypothesis=hypothesis.strip()
                            )
                        )
        tokens = self.tokenizer(texts, truncation=False)["input_ids"]
        if max(map(len, tokens)) > self.context_limit:
            raise ValueError("Openjev pair exceeds context limit; refusing truncation")
        ordered = sorted(enumerate(tokens), key=lambda row: len(row[1]))
        scores = {}
        with self.torch.inference_mode():
            for batch in candidate_batches(ordered, [len(row[1]) for row in ordered], 16, 4096):
                encoded = self.tokenizer.pad(
                    {"input_ids": [row[1] for row in batch]}, padding=True, return_tensors="pt"
                ).to(self.device)
                backbone = getattr(self.engine, self.engine.base_model_prefix)
                hidden = backbone(**encoded, use_cache=False).last_hidden_state
                last = encoded["attention_mask"].sum(-1) - 1
                pooled = hidden[self.torch.arange(len(batch), device=self.device), last]
                values = self.engine.score(pooled).float().softmax(-1)[:, self.entailment]
                for (i, _), value in zip(batch, values, strict=True):
                    scores[i] = value
            flat = self.torch.stack([scores[i] for i in range(len(tokens))])
            outputs = [[] for _ in cases]
            offset = 0
            for index, case_id, q, windows in refs:
                count = len(q.options)
                p = flat[offset : offset + windows * count].reshape(windows, count).max(0).values
                offset += windows * count
                p = (p / p.sum()).cpu().tolist()
                outputs[index].append(
                    Prediction(
                        case_id=case_id,
                        question_id=q.id,
                        probabilities=dict(zip([o.id for o in q.options], p, strict=True)),
                    )
                )
        return outputs
