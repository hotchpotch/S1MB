"""Intern-Decision's released masked-skeleton HF scoring and calibration."""

import importlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .base import decode_answers
from .upstream import UpstreamAdapter
from .vision_native import ordered_images, typed_questions


def intern_questions(case):
    questions = typed_questions(case)
    for key, question in questions.items():
        if question["type"] == "score":
            question["criteria"] = {
                str(option.value): item["description"]
                for option, item in zip(
                    next(q for q in case.questions if q.id == key).options,
                    question["criteria"], strict=True,
                )
            }
    return questions


class InternDecisionAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Intern-Decision context_limit must be positive")
        if sys.version_info < (3, 12):
            raise RuntimeError("The pinned Intern-Decision runtime requires Python 3.12 or newer")
        context_limit = context_limit or 32768
        self.setup("intern-decision", model, revision, source, device)
        native = importlib.import_module("inference")
        self.engine = native.DecisionEngine(
            checkpoint=str(self.path), device=device, max_length=context_limit,
            dtype="bfloat16", attn_implementation="sdpa",
        )
        self.attention_model = self.engine.backend.model
        capacity = self.attention_model.config.get_text_config().max_position_embeddings
        if context_limit > capacity:
            raise ValueError("Intern-Decision context limit exceeds checkpoint capacity")
        self.enable_kernels()
        self.image_receipts = {}
        self.settings.update({
            "dtype": "bfloat16", "max_input_tokens": context_limit,
            "input_length_policy": "reject-overflow-native", "max_candidates": 62,
            "calibration_temperature": self.engine.temperature,
            "calibration_policy": "released-runtime", "renderer": "native-masked-skeleton-v1",
            "media": "ordered-neutral-local-png-paths-native-processor-v1",
        })

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        if any(len(q.options) > 62 for q in case.questions):
            raise ValueError("Intern-Decision supports at most 62 candidates")
        answers = {}
        with TemporaryDirectory(prefix="s1mb-intern-images-") as temporary:
            paths = []
            for index, image in enumerate(images, 1):
                path = Path(temporary) / f"image-{index}.png"
                image.save(path, format="PNG")
                paths.append(str(path))
            for key, question in intern_questions(case).items():
                request = {"state": state if images else case.state, "questions": {"decision": question}}
                if paths:
                    request["images"] = paths
                response = self.engine.predict(request)
                if set(response["answers"]) != {"decision"}:
                    raise ValueError("Intern-Decision returned incorrect answer count")
                answer = response["answers"]["decision"]
                if question["type"] == "score":
                    answer["probabilities"] = {
                        str(index): answer["probabilities"][str(option.value)]
                        for index, option in enumerate(next(q for q in case.questions if q.id == key).options)
                    }
                answers[key] = answer
        return decode_answers(case, answers, anonymous_choice=True)
