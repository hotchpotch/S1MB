"""Decision 3.0's released multimodal runtime and calibrated readout."""

import importlib

from .base import decode_answers
from .upstream import UpstreamAdapter
from .vision_native import ordered_images, typed_questions


class D3Adapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768):
        if context_limit < 1:
            raise ValueError("D3 context_limit must be positive")
        self.setup("d3", model, revision, source, device)
        native = importlib.import_module("d3_runtime")
        self.engine = native.D3.from_pretrained(
            self.path, device=device, batch_size=1, max_length=context_limit, verify="full"
        )
        # The released runtime stores its trained readout as a plain tensor.
        # Expose a storage-sharing parameter view for complete static counts;
        # inference continues to use the untouched native readout tensor.
        self.engine.readout_parameters = self.torch.nn.ParameterList([
            self.torch.nn.Parameter(self.engine.readout, requires_grad=False)
        ])
        self.attention_model = self.engine.backbone
        self.image_receipts = {}
        self.settings.update({
            "dtype": "bfloat16-float32-readout", "batch_size": 1,
            "max_input_tokens": context_limit, "input_length_policy": "reject-overflow-native",
            "media": "native-ordered-pil-images-v1", "calibration_policy": "released-runtime",
            "renderer": "native-structured-anonymous-numeric-score-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
        })

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        answers = {}
        for key, question in typed_questions(case).items():
            response = self.engine.system_one(
                state=state if images else case.state,
                questions={"decision": question}, images=images or None,
            )
            if set(response["answers"]) != {"decision"}:
                raise ValueError("D3 returned incorrect answer count")
            answers[key] = response["answers"]["decision"]
        return decode_answers(case, answers, anonymous_choice=True)
