"""Amos decoder checkpoints through their released in-process decision engine."""

import base64
import importlib
import io
import sys

from .base import decode_answers
from .onejev import onejev_questions
from .upstream import UpstreamAdapter
from .vision_native import ordered_images


class AmosAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768, cuda_shards=1):
        if context_limit < 1:
            raise ValueError("Amos context_limit must be positive")
        if cuda_shards not in {1, 2}:
            raise ValueError("Amos cuda_shards must be 1 or 2")
        self.setup("amos", model, revision, source, device)
        if cuda_shards == 2 and (device != "cuda:0" or self.torch.cuda.device_count() != 2):
            raise ValueError("Amos sharding requires cuda:0 and exactly two visible GPUs")
        if not (self.path / "model" / "config.json").is_file():
            raise ValueError("This bridge requires an Amos decoder checkpoint")
        sys.path.insert(0, str(self.path / "serve"))
        native = importlib.import_module("amos.serve")
        self.torch.backends.cuda.enable_cudnn_sdp(False)
        self.engine = native.load_engine(
            str(self.path), "cpu" if cuda_shards == 2 else "cuda",
            max_len=context_limit, rotations=1,
        )
        capacity = self.engine.model.config.get_text_config().max_position_embeddings
        if context_limit > capacity:
            raise ValueError("Amos context limit exceeds checkpoint capacity")
        if cuda_shards == 2:
            from .cuda_sharding import dispatch_vision_decoder

            self.engine.model.get_output_embeddings().to(device)
            if self.engine.heads is not None:
                self.engine.heads.to(device)
            self.settings["cuda_device_map"] = dispatch_vision_decoder(self.engine.model.model, self.torch)
            self.engine.device = device
            self.engine.autocast = getattr(self.engine.model.config, "amos_export_dtype", None) != "float32"
            if any(p.device.type != "cuda" for p in self.engine.model.parameters()):
                raise RuntimeError("Amos contains non-CUDA weights after dispatch")
            self.settings["weight_loading"] = "cpu-staging-then-cuda-only-inference"
        self.attention_model = self.engine.model
        self.set_attention("sdpa")
        self.enable_kernels()
        self.image_receipts = {}
        self.settings.update({
            "cuda_shards": cuda_shards, "dtype": str(self.engine.model.dtype),
            "max_input_tokens": context_limit, "input_length_policy": "reject-overflow-native",
            "rotations": 1, "calibration_temperatures": self.engine.temperature,
            "cudnn_sdpa": False,
            "calibration_policy": "released-runtime", "candidate_slots": len(self.engine.codec.codes),
            "renderer": "native-structured-anonymous-numeric-score-v1",
            "media": "native-ordered-lossless-png-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
        })

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        encoded = []
        for image in images:
            payload = io.BytesIO()
            image.save(payload, format="PNG")
            encoded.append(base64.b64encode(payload.getvalue()).decode("ascii"))
        answers = {}
        for key, question in onejev_questions(case).items():
            result = self.engine({"state": state if images else case.state,
                                  "images": encoded, "questions": {"decision": question}})
            if set(result["answers"]) != {"decision"}:
                raise ValueError("Amos returned incorrect answer count")
            answers[key] = result["answers"]["decision"]
        return decode_answers(case, answers, anonymous_choice=True)
