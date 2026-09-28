"""Minojev's general checkpoint and typed candidate distributions."""

import importlib
from dataclasses import replace

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches


class MinojevAdapter(UpstreamAdapter):
    def __init__(self, model, revision, source, device, dtype="float32"):
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("Minojev dtype must be float32 or bfloat16")
        self.setup("minojev", model, revision, source, device)
        module = importlib.import_module("minojev.model")
        self.types = importlib.import_module("minojev.types")
        path = self.path / "general" if (self.path / "general/config.json").exists() else self.path
        self.engine = module.DecisionModel.load(path, device=device)
        self.attention_model = self.engine.backbone.model
        if dtype == "bfloat16":
            # Keep the trained decision head in FP32, as required by its native forward path.
            self.engine.backbone.model.to(dtype=self.torch.bfloat16)
        self.options = module.ScoreOptions(mode="fresh", batch_requests=1, device=device)
        native_forward = self.engine.forward_paths

        def bounded_forward(encoded):
            # Bound backbone memory while retaining the head's joint candidate attention.
            return self.torch.cat(
                [
                    native_forward(replace(encoded, paths=batch))
                    for batch in candidate_batches(
                        encoded.paths, [len(path.token_ids) for path in encoded.paths]
                    )
                ],
                dim=0,
            )

        self.engine.forward_paths = bounded_forward
        self.settings = {
            "config": self.engine.config,
            "dtype": str(next(self.engine.backbone.parameters()).dtype),
            "calibration": self.engine.calibration.to_dict(),
            "mode": "fresh",
            "candidate_batch_size": 8,
            "candidate_batch_token_budget": 4096,
            "checkpoint_subdir": str(path.relative_to(self.path)),
        }

    def predict(self, case):
        predictions = []
        for q in case.questions:
            native = questions_for_api([q])
            if q.task == "score":
                native[q.id]["levels"] = native[q.id].pop("criteria")
            request = self.types.request_from_object(
                {"id": case.case_id, "state": case.state, "questions": native}
            )
            row = self.engine.score([request], self.options)[0]
            ids = row["candidate_ids"] if q.task != "score" else [o.id for o in q.options]
            p = dict(zip(ids, row["probabilities"], strict=True))
            # Account for native FP32 softmax error as well as eight-decimal rounding.
            if abs(sum(p.values()) - 1) > len(p) * 0.5e-8 + 1e-6:
                raise ValueError("Minojev distribution exceeds rounding tolerance")
            total = sum(p.values())
            p = {k: v / total for k, v in p.items()}
            check_probabilities(p, [o.id for o in q.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=q.id, probabilities=p))
        return predictions
