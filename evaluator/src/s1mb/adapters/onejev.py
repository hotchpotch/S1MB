"""OneJev's pinned native runtime for complete text and image requests."""

import importlib
import io

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


def onejev_questions(case):
    if any(q.task not in {"noul", "choice", "score"} for q in case.questions):
        raise ValueError("OneJev decision transport does not support Box")
    questions = questions_for_api(case.questions, structured=True, anonymous_choice=True)
    for question in case.questions:
        if question.task == "score":
            item = questions[question.id]
            item["criteria"] = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, item["criteria"], strict=True)
            ]
    return questions


class OneJevAdapter(UpstreamAdapter):
    case_batch_size = 1
    supports_images = True

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("OneJev context_limit must be positive")
        self.setup("onejev", model, revision, source, device)
        self.image_receipts = {}
        native = importlib.import_module("qev.mm_engine")
        self.schema = importlib.import_module("qev.schema")
        calibration_module = importlib.import_module("qev.calibrate")
        calibration_path = self.path / "calibration.json"
        calibration = (
            calibration_module.Calibration.load(calibration_path)
            if calibration_path.exists()
            else calibration_module.Calibration()
        )
        self.engine = native.MMDecisionEngine(
            str(self.path),
            device=device,
            dtype="bfloat16",
            head_dtype="float32",
            calibration=calibration,
            fork_mode="sequential",
            max_branch_tokens=context_limit or 32768,
            max_request_tokens=context_limit or 32768,
            cuda_graphs=False,
            gpu_preprocess=False,
            allow_local_paths=False,
        )
        self.attention_model = self.engine.model
        self.set_attention("sdpa")
        self.enable_kernels()
        self.settings.update(
            {
                "dtype": "bfloat16-float32-head",
                "max_input_tokens": context_limit or 32768,
                "input_length_policy": "reject-overflow-native",
                "fork_mode": "sequential",
                "debias": 1,
                "questions_per_call": 1,
                "media": "native-ordered-pil-images-v1",
                "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
                "native_prompt_version": self.engine.style.name,
                "calibration_temperatures": calibration.temperatures,
                "calibration_policy": "release-file" if calibration_path.exists() else "native-T1",
                "renderer": "native-structured-anonymous-numeric-score-v1",
                "output_rounding": "native-six-decimal-bounded-renormalization",
            }
        )

    def predict(self, case):
        state, media = case.state, []
        if case.images:
            from PIL import Image

            from s1mb.vision import prepare_image

            receipts, references = [], {}
            for index, (name, asset) in enumerate(case.images.items(), 1):
                payload, _, receipt = prepare_image(asset)
                with Image.open(io.BytesIO(payload)) as image:
                    media.append({"type": "image", "image": image.convert("RGB")})
                receipts.append(receipt)
                references[name] = f"<image:{index}>"
            self.image_receipts[case.case_id] = receipts
            state = {"context": case.state, "visual_inputs": references}
        answers = {}
        for key, question in onejev_questions(case).items():
            request = self.schema.SystemOneRequest.model_validate(
                {"state": state, "questions": {"decision": question}}
            )
            response, _ = self.engine.decide(request, debias=1, media=media)
            native = response.model_dump(mode="json")["answers"]
            if set(native) != {"decision"}:
                raise ValueError("OneJev returned incorrect answer count")
            answers[key] = native["decision"]
        return decode_answers(
            case, answers, anonymous_choice=True, rounded=True, rounding_decimals=6
        )
