"""AutoTrust VL's published System 1 choice prompt, LoRA, bias and calibration."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .cuda_sharding import cuda_only_device_map
from .sifr import sifr_questions
from .upstream import UpstreamAdapter, state_text
from .vision_native import ordered_images


def autotrust_choice_prompt(state, definition, image_count):
    """Render the released raw System 1 scaffold with intact typed criteria."""
    values = [state_text(value) for value in definition["criteria"].values()]
    if len(values) > 16:
        raise ValueError("Native trained AutoTrust choice head accepts at most sixteen options")
    images = "<|vision_start|><|image_pad|><|vision_end|>" * image_count
    lines = "\n".join(f"{chr(65 + i)}) {value}" for i, value in enumerate(values))
    return ("[kind] choice\n[state] " + images + state_text(state)
            + "\n[question] " + state_text(definition["instructions"])
            + "\n[options]\n" + lines + "\n[decision]:")


class AutoTrustVisionAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768, cuda_shards=2):
        if cuda_shards != 2:
            raise ValueError("AutoTrust VL requires two CUDA shards")
        self.setup("autotrust-vision", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        config = transformers.AutoConfig.from_pretrained(str(self.path))
        cls = getattr(transformers, config.architectures[0])
        placement = cuda_only_device_map(cls, config, self.torch)
        base = cls.from_pretrained(str(self.path), dtype=self.torch.bfloat16,
                                  device_map=placement, attn_implementation="sdpa").eval()
        peft = importlib.import_module("peft")
        self.engine = peft.PeftModel.from_pretrained(base, str(self.path / "adapter_vllm"),
                                                   is_trainable=False, autocast_adapter_dtype=False).eval()
        if any(parameter.device.type != "cuda" for parameter in self.engine.parameters()):
            raise RuntimeError("AutoTrust contains non-CUDA parameters; CPU offload is disabled")
        self.processor = transformers.AutoProcessor.from_pretrained(str(self.path))
        self.head = json.loads((self.path / "adapter_vllm/decision_head.json").read_text())
        self.temperature = json.loads((self.path / "calibration.json").read_text())["per_kind"]["choice"]
        self.context_limit = context_limit
        self.image_receipts = {}
        self.attention_model = self.engine
        self.settings = {"dtype": "bfloat16", "cuda_shards": 2, "cuda_device_map": placement,
                         "max_input_tokens": context_limit, "input_length_policy": "reject-overflow",
                         "renderer": "released-System1-choice-transport-numeric-score-v1",
                         "typed_mapping": "all-choice-preserving-authored-criteria-and-score-values",
                         "runtime": "transformers-peft-native-protocol-bridge", "thinking": False,
                         "temperature": self.temperature, "calibration_policy": "released-choice-head",
                         "media": "native-ordered-full-images-v1",
                         "image_preprocessing": "shared-full-image-policy-v1-then-native-processor"}
        self.enable_kernels()

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        if images:
            self.image_receipts[case.case_id] = receipts
        definitions = sifr_questions(case)
        predictions = []
        lo, hi = self.head["slots"]["ranges"]["choice"]
        for question in case.questions:
            count = len(question.options)
            if count > hi - lo:
                raise ValueError("AutoTrust trained choice head capacity exceeded; refusing option removal")
            prompt = autotrust_choice_prompt(state if images else case.state, definitions[question.id], len(images))
            inputs = self.processor(text=[prompt], images=images or None, add_special_tokens=False, return_tensors="pt")
            if inputs["input_ids"].shape[1] + 1 > self.context_limit:
                raise ValueError("Expanded AutoTrust input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                output = self.engine(**{key: value.to(self.device) for key, value in inputs.items()},
                                     logits_to_keep=1, use_cache=False)
                ids = self.head["verbalizer_ids"][lo:lo + count]
                logits = output.logits[0, -1, ids].double()
                bias = self.torch.tensor(self.head["bias"][lo:lo + count], device=logits.device, dtype=logits.dtype)
                values = ((logits + bias) / self.temperature).softmax(-1).cpu().tolist()
            probabilities = dict(zip([option.id for option in question.options], values, strict=True))
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id, probabilities=probabilities))
        return predictions
