"""Matilda's released multimodal runtime and calibrated decision head."""

import importlib

from s1mb.data import Prediction, check_probabilities

from .upstream import UpstreamAdapter
from .vision_native import ordered_images, typed_questions


class MatildaAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768, cuda_shards=1):
        if not device.startswith("cuda"):
            raise ValueError("Matilda requires explicit CUDA; no CPU fallback")
        if context_limit < 1:
            raise ValueError("Context limit must be positive")
        self.setup("matilda", model, revision, source, device)
        if cuda_shards not in {1, 2} or (cuda_shards == 2 and self.torch.cuda.device_count() != 2):
            raise ValueError("Matilda requires one or two visible CUDA devices")
        native = importlib.import_module("maincode_jev_serve.model")
        self.decide = importlib.import_module("maincode_jev_serve.decide").decide
        self.engine = native.DecisionModel(
            checkpoint=self.path, device="cpu" if cuda_shards == 2 else device,
            dtype=self.torch.bfloat16, cpu_threads=4,
        )
        self.context_limit = context_limit
        self.image_receipts = {}
        self.settings = {"dtype": "bfloat16", "attention_implementation": "sdpa",
                         "max_input_tokens": context_limit, "cuda_shards": cuda_shards,
                         "input_length_policy": "reject-overflow-native",
                         "temperature": self.engine.temperature, "calibration_policy": "release-file",
                         "case_batch_size": 1, "questions_per_call": 1,
                         "media": "native-ordered-pil-images-v1",
                         "renderer": "native-anonymous-choice-numeric-score-v1",
                         "image_preprocessing": "shared-full-image-policy-v1-then-native-processor"}
        if cuda_shards == 2:
            from .cuda_sharding import dispatch_vision_decoder
            self.engine.readout.to(device)
            self.settings["cuda_device_map"] = dispatch_vision_decoder(self.engine.backbone, self.torch)
            self.engine.device_name = device
            self.settings["weight_loading"] = "cpu-staging-then-cuda-only-inference"
        self.attention_model = self.engine.backbone
        self.enable_kernels()

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        if images:
            self.image_receipts[case.case_id] = receipts
        definitions = typed_questions(case)
        predictions = []
        for question in case.questions:
            definition = definitions[question.id]
            values, _ = self.decide(
                self.engine, state if images else case.state,
                {"decision": definition}, temperature=self.engine.temperature,
                max_tokens=self.context_limit, token_budget=self.context_limit,
                batch_size=1, images=images,
            )
            if set(values) != {"decision"}:
                raise ValueError("Matilda returned incorrect question alignment")
            if question.task == "noul":
                keys = ["false", "true"]
            else:
                keys = [option.id for option in question.options]
            probabilities = dict(zip(keys, values["decision"], strict=True))
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions
