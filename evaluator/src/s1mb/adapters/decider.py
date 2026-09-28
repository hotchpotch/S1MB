"""Decider's typed API without CUDA graphs or shared question interference."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


class DeciderAdapter(UpstreamAdapter):
    def __init__(self, model, revision, source, device):
        self.setup("decider", model, revision, source, device)
        self.engine = importlib.import_module("decider.infer").Decider(
            str(self.path), device=device, dtype=self.torch.bfloat16, use_graphs=False
        )
        self.attention_model = self.engine.m.lm
        self.settings = {
            "dtype": "bfloat16",
            "max_state_tokens": 32768,
            "independent": True,
            "cuda_graphs": False,
        }

    def predict(self, case):
        answers = {}
        for q in case.questions:
            response = self.engine.system_one(
                case.state, questions_for_api([q]), independent=True, max_state_tokens=32768
            )
            answers.update(response["answers"])
        return decode_answers(case, answers, rounded=True)
