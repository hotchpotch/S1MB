"""Vision-Jev's native calibrated heads with authored task and ordered-image prompts."""

import importlib
import json
from types import SimpleNamespace

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, checkpoint_path
from .vision_native import choice_transport, ordered_images, typed_questions


class VisionJevAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, subfolder, base_revision,
                 context_limit=32768):
        self.setup("vision-jev", model, revision, source, device, subfolder=subfolder)
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        tensors = importlib.import_module("safetensors.torch")
        self.native = importlib.import_module("vision_jev.train.rlcd")
        self.sft = importlib.import_module("vision_jev.train.sft")
        adapter_config = json.loads((self.path / "adapter_config.json").read_text())
        base_model = adapter_config["base_model_name_or_path"]
        base_path, resolved = checkpoint_path(base_model, base_revision)
        processor = transformers.AutoProcessor.from_pretrained(str(base_path))
        processor.tokenizer.padding_side = "right"
        backbone = transformers.AutoModelForImageTextToText.from_pretrained(
            str(base_path), dtype=self.torch.bfloat16, attn_implementation="sdpa",
            device_map=device,
        )
        backbone = peft.PeftModel.from_pretrained(backbone, str(self.path)).eval()
        config = json.loads((self.path / "rlcd_config.json").read_text())
        heads = self.native.DecisionHeads(
            backbone.config.text_config.hidden_size, config.get("head_width", 256),
            config.get("head_attention_heads", 4),
        ).to(device).eval()
        heads.load_state_dict(tensors.load_file(str(self.path / "decision_heads.safetensors")))
        self.temperatures = json.loads((self.path / "temperatures.json").read_text())
        self.engine = SimpleNamespace(model=backbone, heads=heads, processor=processor)
        self.extractor = self.native.FeatureExtractor(backbone, processor, config, self.torch.device(device))
        self.attention_model = backbone
        self.enable_kernels()
        self.context_limit = context_limit
        self.image_receipts = {}
        self.settings.update({
            "dtype": "bfloat16-float32-head", "attention_implementation": "sdpa",
            "base_model": base_model, "base_revision": resolved, "subfolder": subfolder,
            "max_input_tokens": context_limit, "input_length_policy": "reject-overflow",
            "media": "native-chat-ordered-pil-extension-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "renderer": "native-heads-authored-task-criteria-v1",
            "prompt_adjustment": "replace-fixed-image-quality-score-rubric-with-authored-criteria",
            "image_token_target": 256, "score_image_token_target": 576,
            "calibration_policy": "released-per-task-temperatures",
            "temperatures": self.temperatures,
        })

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        predictions = []
        for key, question in typed_questions(case).items():
            options = choice_transport(question)
            sample = {"task_type": question["type"], "options": [
                {"id": f"option_{i}", "text": text} for i, text in enumerate(options)
            ]}
            text = "Visible state: " + json.dumps(state if images else case.state, ensure_ascii=False)
            text += "\n\nQuestion: " + question["instructions"]
            text += "\n\nCandidates:\n" + "\n".join(
                f"option_{i}: {value}" for i, value in enumerate(options)
            )
            if question["type"] == "noul":
                text += "\n\nBinary definitions: " + json.dumps(question["criteria"], ensure_ascii=False)
                text += '\nReturn exactly {"noul":true} or {"noul":false}.'
            else:
                text += '\nReturn exactly {"' + question["type"] + '\":\"<candidate_id>\"}.'
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": text})
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.sft.SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]
            target = 576 if question["type"] == "score" else 256
            batch = self.engine.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt", processor_kwargs={"images_kwargs": {
                    "size": {"shortest_edge": 65536, "longest_edge": target * 16 * 16 * 4},
                }},
            ).to(self.device)
            if batch["input_ids"].shape[1] > self.context_limit:
                raise ValueError("Vision-Jev input exceeds context limit")
            with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                output = self.engine.model.base_model.model.model(
                    **batch, return_dict=True, use_cache=False
                )
                evidence = output.last_hidden_state[:, -1]
                candidates = self.extractor._candidate_features(sample)
                mask = self.torch.ones(candidates.shape[:2], dtype=self.torch.bool, device=self.device)
                logits, _ = self.engine.heads(question["type"], evidence, candidates, mask)
                values = self.native.calibrated_probabilities(
                    question["type"], logits, self.temperatures[question["type"]]
                ).float().cpu()
            q = next(q for q in case.questions if q.id == key)
            if q.task == "noul":
                positive = float(values.item())
                probabilities = {"false": 1 - positive, "true": positive}
            else:
                probabilities = dict(zip([o.id for o in q.options], values[0].tolist(), strict=True))
            predictions.append(Prediction(
                case_id=case.case_id, question_id=key, probabilities=probabilities,
            ))
        return predictions
