"""Openvons text decision checkpoints; never initialize an untrained head."""

import importlib
from dataclasses import asdict

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, state_text


class OpenvonsAdapter(UpstreamAdapter):
    def __init__(self, model, revision, source, device):
        self.setup("openvons", model, revision, source, device)
        if not (self.path / "head.pt").is_file():
            raise ValueError("A trained openvons head.pt checkpoint is required")
        self.engine = (
            importlib.import_module("openvons.lm.models.decision_model")
            .DecisionModel.from_checkpoint(str(self.path), device=device)
            .eval()
        )
        self.primitives = importlib.import_module("openvons.core.primitives")
        self.backend = importlib.import_module("openvons.lm.backends.model_backend").ModelBackend(
            self.engine, mode="naive"
        )
        self.settings = {
            "config": asdict(self.engine.cfg),
            "temperature": self.engine.temperature,
            "mode": "naive",
        }

    def predict(self, case):
        out = []
        for q in case.questions:
            opts = (
                q.options if q.task != "noul" else sorted(q.options, key=lambda o: o.id != "true")
            )
            native = self.primitives.Question(
                q.task,
                questions_for_api([q])[q.id]["instructions"],
                [self.primitives.Option(o.id, o.description) for o in opts],
                key=q.id,
            )
            logits = self.backend.logits(state_text(case.state), [native], mode="naive")[0]
            probs = self.engine.probs(logits).float().cpu().tolist()
            out.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities=dict(zip([o.id for o in opts], probs, strict=True)),
                )
            )
        return out
