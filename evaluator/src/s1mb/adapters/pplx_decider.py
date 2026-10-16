"""Perplexity Decider's pinned native calibrated multimodal decision protocol."""

import importlib
import sys

from .base import decode_answers
from .onejev import onejev_questions
from .upstream import UpstreamAdapter
from .vision_native import ordered_images


class PplxDeciderAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768):
        if context_limit < 1:
            raise ValueError("Perplexity Decider context_limit must be positive")
        self.setup("pplx-decider", model, revision, source, device)
        sys.path.insert(0, str(self.path / "source" / "src"))
        native = importlib.import_module("pplx_decider")
        self.engine = native.Decider.from_pretrained(
            self.path, device=device, fast=False, max_length=context_limit
        )
        self.attention_model = self.engine.backbone
        self.enable_kernels()
        self.image_receipts = {}
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa-native-noncausal-masks",
            "max_input_tokens": context_limit, "input_length_policy": "reject-overflow-native",
            "media": "native-ordered-pil-images-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "native_max_pixels": self.engine.max_pixels,
            "calibration_temperature": self.engine.temperature,
            "calibration_policy": "released-temperature-once",
            "renderer": "native-joint-anonymous-numeric-score-v1", "fast_path": False,
        })

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        answers = {}
        for key, question in onejev_questions(case).items():
            result = self.engine.decide({
                "state": state if images else case.state,
                "images": images, "questions": {"decision": question},
            })
            if set(result["answers"]) != {"decision"}:
                raise ValueError("Perplexity Decider returned incorrect answer count")
            answers[key] = result["answers"]["decision"]
        return decode_answers(case, answers, anonymous_choice=True)
