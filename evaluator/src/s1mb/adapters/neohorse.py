"""NeoHorse's bundled typed pointer runtime, with independent bounded calls."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


class NeoHorseAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("neohorse", model, revision, source, device)
        native = importlib.import_module("neohorse_decision")
        self.engine = native.DecisionEngine(
            str(self.path),
            device=device,
            max_state=32768,
            max_branch=32768,
            max_questions=1,
            max_tokens=32768,
        )
        self.attention_model = self.engine.model.lm
        self.settings = {
            "dtype": "bfloat16-backbone-float32-head",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": 32768,
            "case_batch_size": 1,
            "renderer": "native-independent-structured-anonymous-choice-v1",
            "extended_input_condition": "32768 tokens instead of runtime 2048/8192 defaults",
        }
        self.enable_kernels()

    def predict(self, case):
        answers = {}
        for q in case.questions:
            request = questions_for_api([q], structured=True, anonymous_choice=True)
            result = self.engine.predict({"state": case.state, "questions": request})
            answers.update(result["answers"])
        return decode_answers(case, answers, anonymous_choice=True)
