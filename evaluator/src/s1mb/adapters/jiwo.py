"""Pinned jiwo native decoder/readout with calibrated typed probabilities."""

import importlib
import sys

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


def jiwo_questions(case):
    questions = questions_for_api(case.questions, structured=True, anonymous_choice=True)
    for question in case.questions:
        if question.task == "score":
            item = questions[question.id]
            item["criteria"] = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, item["criteria"], strict=True)
            ]
    return questions


class JiwoAdapter(UpstreamAdapter):
    context_limit: int
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("jiwo context_limit must be positive")
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("The pinned jiwo runtime requires Python 3.12")
        self.setup("jiwo", model, revision, source, device)
        self.native = importlib.import_module("jiwo.model")
        self.engine = self.native.DecisionModel.from_pretrained(
            str(self.path), device=device, dtype=self.torch.bfloat16
        )
        self.attention_model = self.engine.backbone
        self.context_limit = context_limit or 32768
        if self.context_limit > self.attention_model.config.max_position_embeddings:
            raise ValueError("jiwo context limit exceeds checkpoint capacity")
        self.set_attention("sdpa")
        self.settings.update({
            "dtype": "bfloat16",
            "max_input_tokens": self.context_limit,
            "batch_tokens": self.context_limit,
            "input_length_policy": "reject-overflow-native",
            "max_candidates": self.engine.config.max_trained_options,
            "temperature": self.engine.config.temperature,
            "type_temperatures": self.engine.config.type_temperatures,
            "questions_per_call": 1,
            "native_python_version": sys.version.split()[0],
            "renderer": "native-structured-anonymous-numeric-score-v1",
        })

    def predict(self, case):
        answers = {}
        for key, question in jiwo_questions(case).items():
            response = self.engine.decide(
                case.state, {"decision": question},
                max_length=self.context_limit, batch_tokens=self.context_limit,
            )
            native = response["answers"]
            if set(native) != {"decision"}:
                raise ValueError("jiwo returned incorrect answer count")
            answers[key] = native["decision"]
        return decode_answers(case, answers, anonymous_choice=True)
