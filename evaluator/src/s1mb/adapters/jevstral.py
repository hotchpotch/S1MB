"""Pinned calibrated Jevstral Stage 4 native pointer-head inference."""

import importlib
import json
import math

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


class JevstralAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("jevstral", model, revision, source, device)
        self.native = importlib.import_module("jevstral.inference")
        encoding = importlib.import_module("jevstral.encode")
        native_model = importlib.import_module("jevstral.model")
        final = self.path / "stage4/final"
        config = json.loads((final / "config.json").read_text())
        if config["base_revision"] != native_model.BASE_REVISION:
            raise ValueError("Jevstral native source and checkpoint base revisions differ")
        limit = encoding.SERVE.max_row if context_limit is None else context_limit
        if not 1 <= limit <= encoding.SERVE.max_row:
            raise ValueError("Jevstral context limit exceeds native serving capacity")
        self.limits = encoding.Limits(min(limit, encoding.SERVE.max_state), limit)
        self.engine = self.native.Predictor(final, path="bf16-merged", cache=False)
        temperature = float(self.engine.model.temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid Jevstral checkpoint temperature")
        self.settings.update({
            "dtype": "bfloat16", "head_dtype": "float32", "checkpoint_subfolder": "stage4/final",
            "base_model": config["base_model"], "base_revision": config["base_revision"],
            "temperature": temperature, "max_input_tokens": limit,
            "max_state_tokens": self.limits.max_state,
            "input_length_policy": "reject-overflow-native",
            "runtime": "native-bf16-merged-fp32-pointer-head",
            "renderer": "native-choice-authored-noul-numeric-score",
            "state_cache": False, "questions_per_call": 1,
        })

    def predict(self, case):
        wire = sifr_questions(case)
        predictions = []
        for question in case.questions:
            request = {"decision": wire[question.id]}
            record = self.native.to_request(case.state, request)
            self.engine.encoder.rows(record, self.limits)
            probabilities = self.engine(case.state, request)
            if set(probabilities) != {"decision"}:
                raise ValueError("Jevstral returned incorrect answer count")
            keys = list(request["decision"]["criteria"])
            values = dict(zip(keys, probabilities["decision"], strict=True))
            check_probabilities(values, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={o.id: values[key] for o, key in
                               zip(question.options, keys, strict=True)},
            ))
        return predictions
