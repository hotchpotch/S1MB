"""Pinned Vev native image and typed-decision inference."""

import base64
import importlib

from .base import decode_answers
from .upstream import UpstreamAdapter
from .vision_native import typed_questions


class VevAdapter(UpstreamAdapter):
    case_batch_size = 1
    supports_images = True

    def __init__(self, model, revision, source, device, context_limit=32768):
        if context_limit < 1:
            raise ValueError("Vev context_limit must be positive")
        self.setup("vev", model, revision, source, device)
        native = importlib.import_module("vev.model")
        self.engine = native.CheckpointEngine(
            str(self.path), dtype="bf16", device=device, attn_implementation="sdpa"
        )
        self.attention_model = self.engine.model.backbone
        self.enable_kernels()
        self.context_limit = context_limit
        self.image_receipts = {}
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa",
            "max_input_tokens": context_limit, "input_length_policy": "reject-overflow-native",
            "media": "native-ordered-state-data-urls-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "max_images": 3, "max_pixels": 1568 * 1568,
            "renderer": "native-structured-anonymous-numeric-score-v1",
            "calibration_policy": "native-allowed-answer-token-softmax",
        })

    def predict(self, case):
        from s1mb.vision import prepare_image

        state = case.state
        if case.images:
            references, receipts = {}, []
            for name, asset in case.images.items():
                payload, mime, receipt = prepare_image(asset)
                references[name] = {"image": {"url": f"data:{mime};base64," + base64.b64encode(payload).decode()}}
                receipts.append(receipt)
            state = {"context": case.state, "visual_inputs": references}
            self.image_receipts[case.case_id] = receipts
        answers = {}
        for key, question in typed_questions(case).items():
            result = self.engine.run(
                state, {"decision": question}, max_images=3, max_pixels=1568 * 1568,
                allow_remote=False, max_state_tokens=self.context_limit,
                max_request_tokens=self.context_limit,
            )
            if set(result.answers) != {"decision"}:
                raise ValueError("Vev returned incorrect answer count")
            answers[key] = result.answers["decision"]
        return decode_answers(case, answers, anonymous_choice=True)
