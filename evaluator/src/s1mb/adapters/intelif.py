"""Pinned Intelif native LoRA merge and anchor readout without truncation."""

import importlib
import json
import sys

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


class IntelifAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Intelif context_limit must be positive")
        if sys.version_info < (3, 12):
            raise RuntimeError("The pinned Intelif runtime requires Python 3.12 or newer")
        self.setup("intelif", model, revision, source, device)
        config = json.loads((self.path / "config.json").read_text())
        limit = context_limit or config["max_tokens"]
        if limit > min(config["max_tokens"], config["base_config"]["context_length"]):
            raise ValueError("Intelif context limit exceeds native release capacity")
        native = importlib.import_module("intelif.model")
        self.engine = native.Intelif.from_pretrained(
            str(self.path), revision=revision, device=device, dtype="bfloat16",
            max_tokens=limit, max_batch_tokens=32768,
        )
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "native-sdpa-gqa",
            "max_input_tokens": limit, "max_batch_tokens": 32768,
            "input_length_policy": "reject-overflow-native", "questions_per_call": 1,
            "base_model": config["base_model"], "base_revision": config["base_revision"],
            "lora": config["lora"], "temperature": 1.0,
            "runtime": "native-merged-lora-anchor-linear-scorer",
            "renderer": "native-choice-authored-noul-numeric-score",
        })

    def predict(self, case):
        predictions = []
        questions = sifr_questions(case)
        for question in case.questions:
            response = self.engine.system_one(case.state, {"decision": questions[question.id]})
            if set(response.answers) != {"decision"}:
                raise ValueError("Intelif returned incorrect answer count")
            probabilities = response.answers["decision"].probabilities
            keys = ([option.id for option in question.options] if question.task == "noul"
                    else [f"option_{i}" for i in range(len(question.options))])
            check_probabilities(probabilities, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={option.id: probabilities[key] for option, key in
                               zip(question.options, keys, strict=True)},
            ))
        return predictions
