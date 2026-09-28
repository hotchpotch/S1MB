"""JevK5's native calibrated letter-logit readout."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


class JevK5Adapter(UpstreamAdapter):
    def __init__(self, model, revision, source, device):
        self.setup("jevk5", model, revision, source, device)
        self.engine = importlib.import_module("jevk5.runtime").JevK5(
            str(self.path), device=device, dtype=self.torch.bfloat16, graphs=False
        )
        self.attention_model = self.engine.model
        self.settings = {
            "dtype": "bfloat16",
            "temperature": self.engine.temperature,
            "large_candidate_policy": "native knockout above 16 candidates",
            "cuda_graphs": False,
        }

    def predict(self, case):
        answers = {
            q.id: self.engine.decide(case.state, questions_for_api([q])[q.id])
            for q in case.questions
        }
        return decode_answers(case, answers, rounded=True)
