"""Visual Jev answer-SFT readout with explicit ordered multi-image transport."""

import importlib
import json
import sys

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, checkpoint_path
from .vision_native import choice_transport, ordered_images, typed_questions


class VisualJevAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, base_revision, context_limit=32768):
        self.setup("visual-jev", model, revision, source, device)
        sys.path.insert(0, str(self.source / "code"))
        native = importlib.import_module("vdm.models.vdm_model")
        self.prompts = importlib.import_module("vdm.models.prompts")
        peft = importlib.import_module("peft")
        base_model = "Qwen/Qwen3-VL-4B-Instruct"
        base, resolved = checkpoint_path(base_model, base_revision)
        self.engine = native.VDM(str(base), device=device, with_heads=False)
        self.engine.backbone = peft.PeftModel.from_pretrained(
            self.engine.backbone, str(self.path)
        ).eval()
        self.attention_model = self.engine.backbone
        self.context_limit = context_limit
        self.image_receipts = {}
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa",
            "base_model": base_model, "base_revision": resolved,
            "max_input_tokens": context_limit, "input_length_policy": "reject-overflow",
            "max_candidates": 16, "native_max_pixels": 200704,
            "media": "ordered-native-image-marker-extension-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "renderer": "native-answer-sft-authored-noul-numeric-score-v1",
            "calibration_policy": "native-allowed-answer-letter-softmax",
        })

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        prefix = self.prompts.prefix_text(json.dumps(state if images else case.state, ensure_ascii=False))
        prefix = prefix.replace(marker, "\n".join(f"{name}:\n{marker}" for name in case.images), 1)
        predictions = []
        for key, question in typed_questions(case).items():
            options = choice_transport(question)
            if not 2 <= len(options) <= 16:
                raise ValueError("Visual Jev requires 2..16 candidates")
            text = prefix + self.prompts.suffix_text(question["instructions"], options)
            inputs = self.engine.processor(
                text=[text], images=images or None, return_tensors="pt",
                images_kwargs={"max_pixels": 200704},
            ).to(self.device)
            if inputs["input_ids"].shape[1] > self.context_limit:
                raise ValueError("Visual Jev input exceeds context limit")
            with self.torch.inference_mode():
                h = self.engine.vl_model(**inputs, use_cache=False).last_hidden_state[:, -1]
                logits = self.torch.nn.functional.linear(
                    h, self.engine.lm_head.weight[self.engine.option_token_ids[:len(options)]]
                )
                values = logits.float().softmax(-1)[0].cpu().tolist()
            q = next(q for q in case.questions if q.id == key)
            keys = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
            predictions.append(Prediction(
                case_id=case.case_id, question_id=key,
                probabilities=dict(zip(keys, values, strict=True)),
            ))
        return predictions
