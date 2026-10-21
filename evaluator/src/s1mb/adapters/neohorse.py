"""NeoHorse's bundled typed pointer runtime, with independent bounded calls."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter
from .vision_native import ordered_images, typed_questions


class NeoHorseAdapter(UpstreamAdapter):
    case_batch_size = 1
    supports_images = True

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
        self.vision_engine = None
        self.image_receipts = {}
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
        if case.images:
            if len(case.images) != 1:
                raise ValueError("NeoHorse's native vision runtime supports exactly one image")
            if self.vision_engine is None:
                native = importlib.import_module("neohorse_decision.vision")
                self.vision_engine = native.VisionDecisionEngine(
                    str(self.path), device=self.device, text_engine=self.engine,
                )
            images, state, receipts = ordered_images(case)
            self.image_receipts[case.case_id] = receipts
            self.settings.update({
                "media": "native-single-image-pil-v1", "max_images": 1,
                "max_visual_input_tokens": 12288,
                "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            })
            answers = {}
            for key, question in typed_questions(case).items():
                response = self.vision_engine.predict(
                    {"state": state, "questions": {"decision": question}}, images[0],
                )
                if set(response["answers"]) != {"decision"}:
                    raise ValueError("NeoHorse returned incorrect question alignment")
                answers[key] = response["answers"]["decision"]
            return decode_answers(case, answers, anonymous_choice=True)
        answers = {}
        for q in case.questions:
            request = questions_for_api([q], structured=True, anonymous_choice=True)
            result = self.engine.predict({"state": case.state, "questions": request})
            answers.update(result["answers"])
        return decode_answers(case, answers, anonymous_choice=True)
