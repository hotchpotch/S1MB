"""Pinned Liquid d1 checkpoints through their bundled one-pass System One runtime."""

import importlib
import importlib.metadata
import json
import os

from s1mb.data import Box, ModelInfo, Prediction, check_probabilities
from s1mb.parameters import parameter_metadata

from .base import score_options
from .upstream import UpstreamAdapter, checkpoint_path

DEFAULT_CONTEXT_LIMIT = 128_000


def text_value(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def d1_question(question) -> dict:
    """Native JSON question with authored text, anonymous Choice keys and ascending Score levels."""
    instructions = "\n\n".join(s for s in [question.system_prompt, question.instructions] if s)
    item = {"type": question.task, "instructions": instructions}
    if question.task == "box":
        return item

    def description(option):
        if option.description_json is not None:
            return text_value(json.loads(option.description_json))
        return option.description

    if question.task == "choice":
        item["criteria"] = {
            f"option_{i}": description(option) for i, option in enumerate(question.options)
        }
    elif question.task == "noul":
        item["criteria"] = {option.id: description(option) for option in question.options}
    else:
        if not 2 <= len(question.options) <= 10:
            raise ValueError("d1 Score requires 2 to 10 native levels")
        item["criteria"] = [description(option) for option in score_options(question, True)]
    return item


def d1_questions(case) -> dict:
    questions = {f"field_{i}": d1_question(q) for i, q in enumerate(case.questions)}
    if any(q.task == "box" for q in case.questions) and len(case.images) != 1:
        raise ValueError("d1 Box requires exactly one image")
    return questions


def decode_answer(case, question, answer, receipts) -> Prediction:
    """Map one native answer to original option IDs or original-image coordinates."""
    if question.task == "box":
        if answer["box"] is None:
            # The model card's documented no-box outcome is scored as zero IoU.
            return Prediction(
                case_id=case.case_id, question_id=question.id, abstention="cannot_determine"
            )
        from s1mb.vision import original_box

        return Prediction(
            case_id=case.case_id,
            question_id=question.id,
            box=original_box(Box.model_validate(answer["box"]), receipts[0]),
        )
    if question.task == "noul":
        yes = float(answer["noul"])
        probabilities = {"false": 1 - yes, "true": yes}
    elif question.task == "score":
        raw = answer["probabilities"]
        if set(raw) != {str(i) for i in range(len(question.options))}:
            raise ValueError("Native Score levels differ from request")
        probabilities = {
            option.id: float(raw[str(i)])
            for i, option in enumerate(score_options(question, True))
        }
    else:
        raw = answer["probabilities"]
        if set(raw) != {f"option_{i}" for i in range(len(question.options))}:
            raise ValueError("Native Choice options differ from request")
        probabilities = {
            option.id: float(raw[f"option_{i}"]) for i, option in enumerate(question.options)
        }
    check_probabilities(probabilities, [option.id for option in question.options])
    return Prediction(case_id=case.case_id, question_id=question.id, probabilities=probabilities)


class D1Adapter(UpstreamAdapter):
    case_batch_size = 1
    supports_images = True

    def __init__(self, model, revision, device, context_limit=None):
        limit = DEFAULT_CONTEXT_LIMIT if context_limit is None else context_limit
        if not 1 <= limit <= DEFAULT_CONTEXT_LIMIT:
            raise ValueError(f"d1 context_limit must be between 1 and {DEFAULT_CONTEXT_LIMIT}")
        device = "cuda:0" if device == "cuda" else device
        path, resolved = checkpoint_path(model, revision)
        self.setup("d1", model, resolved, str(path), device)
        transformers = importlib.import_module("transformers")
        self.engine = transformers.AutoModel.from_pretrained(
            str(path), trust_remote_code=True, dtype=self.torch.bfloat16
        ).to(device)
        # Count before the native runtime fuses projections for inference.
        self.parameters = parameter_metadata(self.engine)
        self.system_one = self.engine.engine
        self.context_limit = limit
        self.image_receipts = {}
        config = self.engine.config
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "d1-sdpa",
            "d1_attention": config.d1_attention,
            "d1_set_size": config.d1_set_size,
            "max_input_tokens": limit,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "questions_per_call": "all-case-questions",
            "renderer": "native-system-one-anonymous-choice-ascending-score-v1",
            "calibration": "native-calibration-json",
            "source_revision": resolved,
            "media": "native-ordered-pil-images-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "native_max_pixels": config.d1_vision_max_pixels,
            "degenerate_box_policy": "retain-zero-area-score-zero-iou",
            "box_missing_policy": "native-found-below-half-counts-as-cannot-determine",
        }

    def predict(self, case):
        questions = d1_questions(case)
        media, receipts = [], []
        if case.images:
            from .vision_native import ordered_images

            media, _, receipts = ordered_images(case)
            self.image_receipts[case.case_id] = receipts
        native = self.system_one
        asked = list(native.named(questions).values())
        sequences = native._sequences([(case.state, asked, media)])
        if max(len(sequence[2]) for sequence in sequences) > self.context_limit:
            raise ValueError(f"d1 input exceeds {self.context_limit} tokens; refusing truncation")
        answers = self.engine.system_one(case.state, questions, media or None)["answers"]
        if set(answers) != set(questions):
            raise ValueError("Native answers differ from request")
        return [
            decode_answer(case, question, answers[key], receipts)
            for key, question in zip(questions, case.questions, strict=True)
        ]

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            **self.parameters,
            id=self.model_id,
            adapter=self.name,
            revision=self.revision,
            settings={
                "device": self.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "source_python_sha256": self.source_digest,
                "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers"]},
                **self.settings,
            },
        )

    def close(self):
        if hasattr(self, "system_one"):
            del self.system_one
        super().close()
