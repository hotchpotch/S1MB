"""Lev native pointer distributions with calibrated, strict input encoding."""

import importlib
import json
import math

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import decision_row
from .upstream import UpstreamAdapter


class LevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("lev", model, revision, source, device)
        self.api = importlib.import_module("lev.api")
        loader = importlib.import_module("lev.evaluate")
        metadata = self.torch.load(self.path / "head.pt", map_location="cpu", weights_only=True)
        if not metadata.get("base_revision"):
            raise ValueError("Lev checkpoint must pin its base model revision")
        self.tokenizer, self.engine = loader.load(
            str(self.path),
            device,
            dtype=self.torch.float32,
        )
        self.attention_model = self.engine.lm
        self.context_limit = context_limit or 8192
        calibration = self.path / "calibration.json"
        self.temperature = (
            json.loads(calibration.read_text()).get("temperature", 1.0)
            if calibration.exists()
            else 1.0
        )
        if (
            not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("Lev temperature must be finite and positive")
        self.settings = {
            "dtype": "float32",
            "temperature": self.temperature,
            "base_model": metadata["base"],
            "base_revision": metadata["base_revision"],
            "max_input_tokens": self.context_limit,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "native-anonymous-choice-numeric-score-v1",
        }

    def predict(self, case):
        predictions = []
        for question in case.questions:
            row = decision_row(case.state, question)
            request = self.api.SystemOneRequest(
                state=row["state"],
                questions={"decision": row["question"]},
            )
            record, _ = self.api.to_record(request)
            encoded = self.engine.encode(
                self.tokenizer,
                record,
                max_state=self.context_limit,
                max_branch=self.context_limit,
                strict=True,
            )
            if len(encoded["ids"]) > self.context_limit:
                raise ValueError("Lev input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                logits = self.engine.forward(encoded)[0]
                values = (logits.float() / self.temperature).softmax(-1).cpu().tolist()
            ids = ["false", "true"] if question.task == "noul" else [o.id for o in question.options]
            probabilities = dict(zip(ids, values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=question.id,
                    probabilities=probabilities,
                )
            )
        return predictions
