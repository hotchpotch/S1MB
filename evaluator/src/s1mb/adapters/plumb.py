"""Plumb's released single-read, calibrated JevK5 decision protocol."""

import importlib

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import decision_row
from .jevlite import extended_labels
from .upstream import UpstreamAdapter


class PlumbAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("plumb", model, revision, source, device)
        self.native = importlib.import_module("jevk5.runtime")
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        labels = extended_labels(tokenizer, list(self.native.LETTERS), max_candidates or 16)
        self.native.__dict__["LETTERS"] = labels
        self.engine = self.native.JevK5(
            str(self.path), device=device, dtype=self.torch.bfloat16, graphs=False
        )
        self.context_limit = context_limit or 32768
        self.capacity = len(labels)
        self.attention_model = self.engine.model
        self.settings = {
            "dtype": "bfloat16",
            "max_input_tokens": self.context_limit,
            "max_candidates": self.capacity,
            "native_max_candidates": 16,
            "answer_codes": labels,
            "temperature": self.engine.temperature,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "native-single-read-anonymous-choice-numeric-score-v1",
        }
        self.set_attention("sdpa")
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for question in case.questions:
            row = decision_row(case.state, question)
            definition = row["question"]
            options = self.native.decision_options(definition)
            if len(options) > self.capacity:
                raise ValueError("Plumb candidate capacity exceeded")
            ids = self.engine.encode(
                row["state"], definition["instructions"], [text for _, text in options]
            )
            if len(ids) > self.context_limit:
                raise ValueError("Plumb input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                logits = self.engine.letter_logits(ids, len(options))
            values = self.torch.as_tensor(logits / self.engine.temperature).softmax(-1).tolist()
            raw = dict(zip([key for key, _ in options], values, strict=True))
            keys = [o.id for o in question.options] if question.task == "noul" else list(raw)
            probabilities = {o.id: raw[key] for o, key in zip(question.options, keys, strict=True)}
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
