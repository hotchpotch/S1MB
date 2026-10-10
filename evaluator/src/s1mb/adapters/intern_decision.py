"""Pinned Intern-Decision masked-symbol scoring with native calibration."""

import importlib
import sys

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


def intern_questions(case):
    questions = questions_for_api(case.questions, structured=True, anonymous_choice=True)
    for question in case.questions:
        if question.task == "score":
            item = questions[question.id]
            item["criteria"] = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, item["criteria"], strict=True)
            ]
    return questions


class InternDecisionAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Intern-Decision context_limit must be positive")
        if sys.version_info < (3, 12):
            raise RuntimeError("The pinned Intern-Decision runtime requires Python 3.12 or newer")
        self.setup("intern-decision", model, revision, source, device)
        self.native = importlib.import_module("inference")
        self.context_limit = context_limit or 32768
        self.engine = self.native.DecisionEngine(
            checkpoint=str(self.path), max_length=self.context_limit,
            device=device, dtype="bfloat16", attn_implementation="sdpa",
        )
        self.attention_model = self.engine.backend.model
        capacity = self.attention_model.config.get_text_config().max_position_embeddings
        if self.context_limit > capacity:
            raise ValueError("Intern-Decision context limit exceeds checkpoint capacity")
        self.settings.update({
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit,
            "max_candidates": len(self.native.ANSWER_SYMBOLS),
            "input_length_policy": "reject-overflow-native",
            "candidate_capacity_policy": "reject-overflow-native-no-filtering",
            "temperature": self.engine.temperature,
            "readout": "native-logits-before-trained-decision-marker",
            "questions_per_call": 1,
            "native_python_version": sys.version.split()[0],
            "renderer": "native-structured-anonymous-numeric-score-v1",
        })

    def predict(self, case):
        answers = {}
        for key, question in intern_questions(case).items():
            response = self.engine.predict({"state": case.state, "questions": {"decision": question}})
            native = response["answers"]
            if set(native) != {"decision"}:
                raise ValueError("Intern-Decision returned incorrect answer count")
            answers[key] = native["decision"]
        return decode_answers(case, answers, anonymous_choice=True)
