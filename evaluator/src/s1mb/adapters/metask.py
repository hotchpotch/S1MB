"""Metask's native schema prompt and task-specific calibrated code logits."""

import importlib
import json
import string

from s1mb.data import Prediction, check_probabilities

from .jevlite import extended_labels
from .nimble import nimble_field
from .upstream import UpstreamAdapter, state_text


class MetaskAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("metask", model, revision, source, device)
        self.native = importlib.import_module("jev_schema")
        loader = importlib.import_module("jev_scorer")
        self.engine, self.tokenizer, _ = loader.load_model(
            str(self.path), device=(device, self.torch.bfloat16)
        )
        self.attention_model = self.engine
        self.context_limit = context_limit or 4096
        self.codes = extended_labels(
            self.tokenizer, list(string.ascii_uppercase), max_candidates or 26
        )
        self.temperature = json.loads((self.path / "temperature.json").read_text())["by_kind"]
        self.settings = {
            "dtype": "bfloat16",
            "max_input_tokens": self.context_limit,
            "max_candidates": len(self.codes),
            "native_max_candidates": 26,
            "answer_codes": self.codes,
            "temperature": self.temperature,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "native-enum-authored-noul-numeric-score-v1",
        }
        self.set_attention("sdpa")
        self.enable_kernels()

    def prepare(self, state, question):
        keys, schema = nimble_field(question)
        if len(keys) > len(self.codes):
            raise ValueError("Metask candidate capacity exceeded; refusing to drop options")
        if len(keys) <= 26:
            prepared = self.native.prepare_prompts(
                self.tokenizer, state_text(state), schema, self.context_limit
            )
            return keys, prepared.full_ids[0], prepared.candidate_ids[0]
        # Preserve the native format while explicitly extending its answer codes.
        field = schema["decision"]
        choices = [
            {"code": code, "value": key, "description": field["choice_descriptions"][key]}
            for code, key in zip(self.codes, keys)
        ]
        content = (
            self.native.safe_json(
                {
                    "context": state_text(state),
                    "schema": [
                        {
                            "name": "decision",
                            "description": field["description"],
                            "choices": choices,
                        }
                    ],
                }
            )
            + '\n\nRequested field: "decision"'
        )
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.native.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.context_limit:
            raise ValueError("Metask input exceeds context limit; refusing truncation")
        candidates = []
        for code in self.codes[: len(keys)]:
            combined = self.tokenizer.encode(prompt + code, add_special_tokens=False)
            if (
                combined[: len(ids)] != ids
                or len(combined) != len(ids) + 1
                or combined[-1] in self.tokenizer.all_special_ids
            ):
                raise ValueError("Metask code must be one ordinary token at the answer boundary")
            candidates.append(combined[-1])
        if len(set(candidates)) != len(candidates):
            raise ValueError("Metask answer codes must be distinct")
        return keys, ids, candidates

    def predict(self, case):
        predictions = []
        for question in case.questions:
            _keys, ids, candidates = self.prepare(case.state, question)
            with self.torch.inference_mode():
                logits = (
                    self.engine(
                        input_ids=self.torch.tensor([ids], device=self.device),
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    .logits[0, -1]
                    .float()
                )
                values = (
                    (logits[candidates] / self.temperature[question.task])
                    .softmax(-1)
                    .cpu()
                    .tolist()
                )
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
