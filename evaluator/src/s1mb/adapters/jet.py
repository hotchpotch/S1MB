"""Jet native FLA and restricted-label readout with complete authored criteria."""

import importlib
import json
import sys
from typing import Any

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


def jet_questions(case):
    questions = sifr_questions(case)
    for item in questions.values():
        if not isinstance(item["instructions"], str):
            item["instructions"] = json.dumps(item["instructions"], ensure_ascii=False)
        item["criteria"] = {key: value if isinstance(value, str) else
                            json.dumps(value, ensure_ascii=False)
                            for key, value in item["criteria"].items()}
    return questions


class JetAdapter(UpstreamAdapter):
    case_batch_size = 1
    native_format: Any
    native_inference: Any
    native_runtime: Any

    def __init__(self, model, revision, source, device, context_limit=None):
        if sys.version_info < (3, 12):
            raise RuntimeError("Jet requires Python 3.12 or newer")
        limit = 16384 if context_limit is None else context_limit
        if not 0 < limit <= 16384:
            raise ValueError("Jet context limit must fit the native 16384-token budget")
        self.setup("jet", model, revision, source, device)
        native = importlib.import_module("jet")
        self.engine = native.Jet(str(self.path), max_tokens=limit)
        self.native_format = importlib.import_module("format")
        self.native_inference = importlib.import_module("inference")
        self.native_runtime = importlib.import_module("runtime")
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "native-sdpa-fla",
            "max_input_tokens": limit, "input_length_policy": "reject-overflow-native",
            "temperature_by_native_type": self.engine.temperatures,
            "questions_per_call": 1, "max_candidates": 255,
            "renderer": "native-choice-authored-noul-numeric-score",
            "runtime": "released-fla-gated-delta-pytorch-convolution-restricted-label-head",
            "probability_readout": "native-float64-softmax-before-display-rounding",
        })

    def predict(self, case):
        predictions = []
        questions = jet_questions(case)
        for question in case.questions:
            native_question = self.native_format.Question.from_dict(questions[question.id])
            ids = self.native_inference.encode(
                self.engine.tokenizer, case.state, native_question, 10**9,
            )
            if len(ids) > self.engine.max_tokens:
                raise ValueError("Jet complete prompt exceeds the native token budget")
            labels = self.native_format.label_token_ids(self.engine.tokenizer, native_question)
            torch = importlib.import_module("torch")
            with torch.no_grad():
                logits = self.native_runtime.label_logits(
                    self.engine.model, {"ids": ids, "labels": labels},
                )
                values = (logits.double() / self.engine.temperatures[native_question.type]).softmax(-1)
                probabilities = dict(zip(native_question.keys, values.cpu().tolist(), strict=True))
            keys = ([o.id for o in question.options] if question.task == "noul"
                    else [f"option_{i}" for i in range(len(question.options))])
            check_probabilities(probabilities, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={o.id: probabilities[key] for o, key in
                               zip(question.options, keys, strict=True)},
            ))
        return predictions
