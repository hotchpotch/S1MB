"""Pinned Jev-Style v3 native rendering, calibrated readout and option chunks."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter


def jev_style_questions(case):
    questions = questions_for_api(case.questions, structured=True, anonymous_choice=True)
    result = {}
    for question in case.questions:
        item = questions[question.id]
        instructions = item["instructions"]
        criteria = item["criteria"]
        if question.task == "score":
            criteria = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, criteria, strict=True)
            ]
        result[question.id] = {
            "t": question.task,
            "ins": instructions if isinstance(instructions, str) else json.dumps(
                instructions, ensure_ascii=False
            ),
            "crit": criteria,
        }
    return result


class JevStyleAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Jev-Style context_limit must be positive")
        self.setup("jev-style", model, revision, source, device)
        self.native = importlib.import_module("jev_style_decision")
        self.engine = self.native.JevStyleDecision(
            str(self.path),
            device="cuda",
            dtype="float32",
            category=None,
            max_len=context_limit or self.native.CONTEXT_LIMIT,
            verify=True,
            split_options=True,
            cuda_graphs=False,
        )
        self.attention_model = self.engine.model
        self.set_attention("sdpa")
        self.settings.update(
            {
                "dtype": "float32",
                "max_input_tokens": context_limit or self.native.CONTEXT_LIMIT,
                "head_max_tokens": self.native.HARD_HEAD_MAX,
                "input_length_policy": "reject-overflow-native",
                "option_chunk_policy": "native-lossless-option-chunks",
                "calibration_policy": "native-global-no-category",
                "native_temperatures": self.engine.temperatures,
                "cuda_graphs": False,
                "renderer": "native-structured-anonymous-numeric-score-v1",
            }
        )

    def predict(self, case):
        questions = jev_style_questions(case)
        predictions = []
        for question in case.questions:
            response = self.engine.decide(case.state, questions[question.id])
            probabilities = response["probabilities"]
            keys = (
                [f"option_{i}" for i in range(len(question.options))] if question.task == "choice"
                else [str(i) for i in range(len(question.options))] if question.task == "score"
                else [option.id for option in question.options]
            )
            check_probabilities(probabilities, keys)
            aligned = {
                option.id: probabilities[key]
                for option, key in zip(question.options, keys, strict=True)
            }
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=aligned
            ))
        return predictions
