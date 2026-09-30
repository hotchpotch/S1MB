"""Pinned Lumma-Fev inference with native probabilities and strict admission."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


class LummaAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("lumma", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        self.engine = (
            transformers.AutoModel.from_pretrained(
                str(self.path), trust_remote_code=True, dtype="auto"
            )
            .to(device)
            .eval()
        )
        self.native = importlib.import_module(type(self.engine).__module__)
        self.attention_model = self.engine.lm
        self.set_attention("sdpa")
        self.settings.update(
            {
                "dtype": "checkpoint-native",
                "input_length_policy": "reject-overflow",
                "state_and_row_limits": list(self.engine.limits()),
                "temperature": self.engine.config.temperature,
                "case_batch_size": 1,
                "renderer": "native-structured-anonymous-choice-v1",
            }
        )

    def predict(self, case):
        engine = self.engine
        state_ids = engine.text_ids(engine.tokenizer, self.native.render(case.state))
        if len(state_ids) + 1 > engine.limits()[0]:
            raise ValueError("Lumma state exceeds native limit; refusing truncation")
        request = questions_for_api(case.questions, structured=True, anonymous_choice=True)
        answers = engine.decide(case.state, request)
        return decode_answers(case, answers, anonymous_choice=True, rounded=True)
