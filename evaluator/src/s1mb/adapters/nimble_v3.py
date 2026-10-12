"""Bespoke Nimble v3 native calibrated-contract candidate scoring."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


class NimbleV3Adapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        limit = 32768 if context_limit is None else context_limit
        if not 1 <= limit <= 262144:
            raise ValueError("Nimble v3 context limit must be within 1..262144")
        self.setup("nimble-v3", model, revision, source, device)
        self.native = importlib.import_module("nimble.evaluation.decision_index_engine")
        contract = json.loads((self.path / "schema_config.json").read_text())
        self.engine = self.native.NimbleEngine(
            adapter=str(self.path), model=contract["model"], revision=contract["revision"],
            max_tokens=limit, batch_tokens=limit, shared_prefix=False,
        )
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa",
            "base_model": contract["model"], "base_revision": contract["revision"],
            "max_input_tokens": limit, "input_length_policy": "reject-overflow-native",
            "max_candidates": 255, "temperature": 1.0,
            "calibration": "native-v3-unscaled-candidate-softmax",
            "renderer": "native-choice-authored-noul-numeric-score",
            "shared_prefix": False, "questions_per_call": 1,
        })

    def predict(self, case):
        wire = sifr_questions(case)
        predictions = []
        for question in case.questions:
            request = {"decision": wire[question.id]}
            response, _ = self.engine(case.state, request)
            answers = response["answers"]
            if set(answers) != {"decision"}:
                raise ValueError("Nimble v3 returned incorrect answer count")
            probabilities = answers["decision"]["probabilities"]
            keys = list(request["decision"]["criteria"])
            check_probabilities(probabilities, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={o.id: probabilities[key] for o, key in
                               zip(question.options, keys, strict=True)},
            ))
        return predictions
