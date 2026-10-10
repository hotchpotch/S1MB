"""Torchcast's native causal decoder, typed calibration, and wide-choice tournament."""

import importlib
import math
import os

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


class TorchcastAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("torchcast", model, revision, source, device)
        self.native = importlib.import_module("torchcast_decision.model")
        previous = os.environ.get("STARTLUX_ALLOW_SLOW")
        # The released probe predates Transformers' decorated kernel integration.
        # Permit construction, then explicitly activate supported GPU kernels.
        os.environ["STARTLUX_ALLOW_SLOW"] = "1"
        try:
            self.engine = self.native.TorchcastDecision(
                str(self.path), device=device, max_length=32768,
                max_batch_tokens=32768, graphs=False, images=False,
            )
        finally:
            if previous is None:
                os.environ.pop("STARTLUX_ALLOW_SLOW", None)
            else:
                os.environ["STARTLUX_ALLOW_SLOW"] = previous
        if any(not math.isfinite(t) or t <= 0 for t in self.engine.temperature.values()):
            raise ValueError("Invalid native Torchcast temperature")
        self.attention_model = self.engine.body
        self.set_attention("sdpa")
        self.enable_kernels()
        # Native inference retains only selected output rows as a plain tensor.
        # Register their existing storage so metadata includes the complete readout.
        self.engine.readout_metadata = self.torch.nn.Module()
        self.engine.readout_metadata.register_parameter(
            "weight", self.torch.nn.Parameter(self.engine.letter_rows, requires_grad=False),
        )
        self.settings.update({
            "dtype": "bfloat16-backbone-float32-letter-readout",
            "max_input_tokens": 32768,
            "input_length_policy": "native-reject-overflow-every-tournament-row",
            "temperatures": dict(self.engine.temperature),
            "wide_choice": {
                "group": self.engine.group, "keep": self.engine.keep,
                "residual": self.engine.residual,
            },
            "typed_mapping": "anonymous-choice-authored-order-numeric-score-original-task-temperature",
            "renderer": "native-StartLux-Decision-v1",
            "cuda_graphs": False,
            "images": False,
            "case_batch_size": 1,
        })

    def predict(self, case):
        questions = sifr_questions(case)
        predictions = []
        for question in case.questions:
            temperature = self.engine.temperature["choice"]
            self.engine.temperature["choice"] = self.engine.temperature[question.task]
            try:
                answers, _ = self.engine.decide(
                    case.state, {question.id: questions[question.id]},
                )
            finally:
                self.engine.temperature["choice"] = temperature
            if set(answers) != {question.id}:
                raise ValueError("Torchcast returned unexpected questions")
            raw = answers[question.id]["probabilities"]
            keys = [
                option.id if question.task == "noul" else f"option_{i}"
                for i, option in enumerate(question.options)
            ]
            check_probabilities(raw, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={
                    option.id: raw[key]
                    for option, key in zip(question.options, keys, strict=True)
                },
            ))
        return predictions
