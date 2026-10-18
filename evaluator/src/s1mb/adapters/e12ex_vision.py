"""Released e12ex2 vision LoRA with native option logits and complete criteria."""

import base64
import importlib
import io
import json

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, checkpoint_path, source_path
from .vision_native import ordered_images


class E12exVisionAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, base_revision, context_limit=32768, cuda_shards=2):
        if context_limit < 1:
            raise ValueError("e12ex2 context_limit must be positive")
        if not base_revision:
            raise ValueError("e12ex2 requires an explicit base revision")
        if cuda_shards != 2 or device != "cuda:0":
            raise ValueError("e12ex2 requires cuda:0 and two CUDA shards")
        self.setup("e12ex-vision", model, revision, source, device)
        if self.torch.cuda.device_count() != 2:
            raise ValueError("e12ex2 requires exactly two visible CUDA devices")
        config = json.loads((self.path / "adapter_config.json").read_text())
        base = config["base_model_name_or_path"]
        base_path, resolved_base = checkpoint_path(base, base_revision)
        if (self.path / "head.pt").exists():
            raise ValueError("This bridge requires the released head-free vision adapter")
        _, native_digest = source_path(str(self.path / "code"))
        native = importlib.import_module("serve.engine")
        self.format = importlib.import_module("jv.fmt")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        processor = transformers.AutoProcessor.from_pretrained(str(base_path), max_pixels=4_000_000)
        backbone = transformers.AutoModelForImageTextToText.from_pretrained(
            str(base_path), dtype=self.torch.bfloat16, device_map={"": "cpu"},
            attn_implementation="sdpa",
        ).eval()
        if context_limit > backbone.config.get_text_config().max_position_embeddings:
            raise ValueError("e12ex2 context limit exceeds checkpoint capacity")
        adapted = peft.PeftModel.from_pretrained(backbone, str(self.path)).eval()
        from .cuda_sharding import dispatch_vision_decoder

        adapted.get_output_embeddings().to(device)
        placement = dispatch_vision_decoder(backbone.model, self.torch)
        backbone.hf_device_map = {**{f"model.{name}": gpu for name, gpu in placement.items()}, "lm_head": 0}
        # Merge on CUDA, as in the released loader, after positioning every LoRA layer.
        merged = adapted.merge_and_unload().eval()
        accelerate = importlib.import_module("accelerate")
        accelerate.hooks.remove_hook_from_submodules(merged.model)
        accelerate.dispatch_model(merged.model, device_map=placement, main_device=0, force_hooks=True)
        if any(parameter.device.type != "cuda" for parameter in merged.parameters()):
            raise RuntimeError("e12ex2 contains non-CUDA weights after dispatch")
        # Only weight loading differs from the released constructor's single-GPU placement.
        # All rendering, label lookup, model forwards and probability computations stay native.
        self.engine = native.JVEngine.__new__(native.JVEngine)
        native.Engine.__init__(self.engine)
        self.engine.torch, self.engine.proc, self.engine.model = self.torch, processor, merged
        self.engine.head, self.engine.perms, self.engine.limit = None, 1, context_limit
        self.attention_model = merged
        self.enable_kernels()
        self.image_receipts = {}
        self.settings.update({
            "base_model": base, "base_revision": resolved_base, "cuda_shards": 2,
            "cuda_device_map": placement, "dtype": "bfloat16",
            "native_checkpoint_python_sha256": native_digest,
            "weight_loading": "cpu-staging-then-cuda-lora-merge-and-cuda-only-inference",
            "max_input_tokens": context_limit, "input_length_policy": "reject-overflow-native",
            "native_max_pixels": 4_000_000, "option_orders": 1,
            "candidate_slots": len(self.format.LABELS), "calibration_policy": "native-identity-temperature",
            "trained_candidate_slots": 62,
            "renderer": "native-choice-authored-noul-numeric-score-v1",
            "typed_mapping": "all-primitives-explicit-choice-criteria",
            "media": "native-ordered-lossless-png-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
        })

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        parts = [state if images else case.state]
        for image in images:
            payload = io.BytesIO()
            image.save(payload, format="PNG")
            parts.append({"image": "data:image/png;base64," + base64.b64encode(payload.getvalue()).decode("ascii")})
        questions = sifr_questions(case)
        predictions = []
        for question in case.questions:
            response, _ = self.engine(parts if images else case.state, {"decision": questions[question.id]})
            if set(response["answers"]) != {"decision"}:
                raise ValueError("e12ex2 returned incorrect answer count")
            probabilities = response["answers"]["decision"]["probabilities"]
            keys = ([option.id for option in question.options] if question.task == "noul"
                    else [f"option_{i}" for i in range(len(question.options))])
            check_probabilities(probabilities, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={option.id: probabilities[key] for option, key in zip(question.options, keys, strict=True)},
            ))
        return predictions
