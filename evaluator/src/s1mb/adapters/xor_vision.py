"""Xor's released two-order decision protocol on CUDA-only Transformers."""

import base64
import importlib
import importlib.util
import io
import json
import os
from typing import Any, cast

from s1mb.data import Prediction, check_probabilities

from .cuda_sharding import cuda_only_device_map
from .sifr import sifr_questions
from .upstream import UpstreamAdapter
from .vision_native import ordered_images


class XorVisionAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768, cuda_shards=2):
        if cuda_shards != 2:
            raise ValueError("Xor vision requires two CUDA shards")
        self.setup("xor-vision", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        config = transformers.AutoConfig.from_pretrained(str(self.path))
        cls = getattr(transformers, config.architectures[0])
        activation_reserve = 5 * 1024**3
        placement = cuda_only_device_map(cls, config, self.torch, activation_reserve=activation_reserve)
        self.engine = cls.from_pretrained(str(self.path), dtype=self.torch.bfloat16,
                                         device_map=placement, attn_implementation="sdpa").eval()
        self.processor = transformers.AutoProcessor.from_pretrained(str(self.path))
        self.context_limit = context_limit
        self.image_receipts = {}
        spec = importlib.util.spec_from_file_location("_s1mb_xor_vision_native", self.source / "xn_base_server.py")
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load Xor's native serving source")
        self.native = cast(Any, importlib.util.module_from_spec(spec))
        values = {"XN_MODEL_PATH": str(self.path), "XN_MAX_CONTEXT": str(context_limit),
                  "XN_TEMP_JSON": json.dumps({"choice": 1.95, "noul": 1.0, "score": 1.0})}
        previous = {key: os.environ.get(key) for key in values}
        try:
            os.environ.update(values)
            spec.loader.exec_module(self.native)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.native.generate = self.generate
        self.attention_model = self.engine
        self.settings = {"dtype": "bfloat16", "cuda_shards": 2, "cuda_device_map": placement,
                         "cuda_activation_reserve_bytes": activation_reserve,
                         "max_input_tokens": context_limit, "input_length_policy": "reject-overflow",
                         "renderer": "native-xor-two-order-anonymous-choice-numeric-score-v1",
                         "runtime": "transformers-native-protocol-bridge", "two_order": True,
                         "temperature_by_type": self.native.TEMP,
                         "media": "native-ordered-full-images-v1",
                         "image_preprocessing": "shared-full-image-policy-v1-then-native-processor"}

    def generate(self, rows, label_sets, images=None, video=None):
        if video is not None:
            raise ValueError("Video transport is unsupported")
        prepared = []
        for row in rows:
            if self.current_images:
                text = self.native.tok.decode(row, skip_special_tokens=False)
                if self.native.E(text) != row:
                    raise ValueError("Native Xor prompt cannot round-trip through its processor")
                batch = self.processor(text=[text], images=self.current_images,
                                       add_special_tokens=False, return_tensors="pt")
            else:
                batch = {"input_ids": self.torch.tensor([row])}
            if batch["input_ids"].shape[1] + 1 > self.context_limit:
                raise ValueError("Expanded Xor input exceeds context limit; refusing truncation")
            prepared.append(batch)
        results = []
        for batch, labels in zip(prepared, label_sets, strict=True):
            with self.torch.inference_mode():
                output = self.engine(**{key: tensor.to(self.device) for key, tensor in batch.items()},
                                     logits_to_keep=1, use_cache=False)
                values = output.logits[0, -1].float().log_softmax(-1)[labels].cpu().tolist()
            results.append({"meta_info": {"prompt_tokens": batch["input_ids"].shape[1],
                "output_token_ids_logprobs": [[[value, label, ""] for value, label in zip(values, labels, strict=True)]]}})
        return results

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.current_images = images
        uris = []
        for image in images:
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            uris.append("data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii"))
        if images:
            self.image_receipts[case.case_id] = receipts
        predictions = []
        wire = sifr_questions(case)
        previous = os.environ.get("XN_TWO_ORDER")
        try:
            os.environ["XN_TWO_ORDER"] = "1"
            for question in case.questions:
                answers, _, _ = self.native.evaluate(state if images else case.state,
                                                     {"decision": wire[question.id]}, images=uris or None)
                if set(answers) != {"decision"}:
                    raise ValueError("Xor returned incorrect answer alignment")
                probabilities = answers["decision"]["probabilities"]
                keys = list(wire[question.id]["criteria"])
                check_probabilities(probabilities, keys)
                predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                    probabilities={option.id: probabilities[key] for option, key in zip(question.options, keys, strict=True)}))
        finally:
            if previous is None:
                os.environ.pop("XN_TWO_ORDER", None)
            else:
                os.environ["XN_TWO_ORDER"] = previous
            self.current_images = []
        return predictions
