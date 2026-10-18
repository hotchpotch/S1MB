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

    def __init__(self, model, revision, source, device, context_limit=32768, cuda_shards=1):
        if context_limit < 1:
            raise ValueError("Perplexity Decider context_limit must be positive")
        if cuda_shards not in {1, 2}:
            raise ValueError("Perplexity Decider cuda_shards must be 1 or 2")
        self.setup("pplx-decider", model, revision, source, device)
        if cuda_shards == 2 and (device != "cuda:0" or self.torch.cuda.device_count() != 2):
            raise ValueError("Perplexity Decider sharding requires cuda:0 and exactly two visible GPUs")
        sys.path.insert(0, str(self.path / "source" / "src"))
        native = importlib.import_module("pplx_decider")
        self.engine = native.Decider.from_pretrained(
            self.path, device="cpu" if cuda_shards == 2 else device,
            fast=False, max_length=context_limit
        )
        if cuda_shards == 2:
            from .cuda_sharding import dispatch_vision_decoder

            # Native CPU loading uses FP32; restore the native CUDA BF16 dtype before dispatch.
            self.engine.backbone.to(dtype=self.torch.bfloat16)
            self.settings["cuda_device_map"] = dispatch_vision_decoder(self.engine.backbone, self.torch)
            self.engine.readout.to(device=device, dtype=self.torch.bfloat16)
            self.engine.device = self.torch.device(device)
            self.settings["weight_loading"] = "cpu-float32-staging-then-cuda-bfloat16-only-inference"
        self.settings["cuda_shards"] = cuda_shards
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
