"""Shisa DE-2's released single-image SDK readout, without top-k loss."""

import importlib

from s1mb.data import Prediction, check_probabilities

from .rune import RuneAdapter
from .sifr import sifr_questions
from .upstream import state_text
from .vision_native import ordered_images


class ShisaVisionAdapter(RuneAdapter):
    def __init__(self, model, revision, source, device, context_limit=32768, cuda_shards=2):
        super().__init__(model, revision, source, device, context_limit, cuda_shards)
        native = importlib.import_module("shisa_de.readout")
        self.schema = importlib.import_module("shisa_de.questions")
        self.native = native.Readout("http://127.0.0.1", model, tokenizer=str(self.path), local_files_only=True)
        self.native._tokenizer = self.tokenizer
        self.name = "shisa-vision"
        self.settings.update({
            "renderer": "native-shisa-sdk-choice-transport-single-image-numeric-score-v1",
            "inference_policy": "released-SDK-direct-image-read-not-index-repeat-policy",
            "temperature": 1.0, "calibration_policy": "native-images-uncalibrated-default",
            "max_images": 1, "candidate_readout": "complete-restricted-logits-no-topk-omission",
        })

    def predict(self, case):
        if len(case.images) > 1:
            raise ValueError("Shisa's SDK image readout supports at most one image")
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        definitions = sifr_questions(case)
        predictions = []
        for question in case.questions:
            spec = definitions[question.id]
            item = self.schema.Choice(instructions=state_text(spec["instructions"]), criteria=spec["criteria"])
            if images:
                payload, options = self.native._payload(state, item)
                # The SDK's image scaffold contains one image followed by the JSON payload.
                messages = self.native._messages(payload, "data:image/png;base64,unused")
                messages[1]["content"][0] = {"type": "image", "image": images[0]}
                inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_dict=True, return_tensors="pt")
                slots = self.native.slots(len(options))
            else:
                # DE-2's direct text policy admits its full single-token codebook.
                codes = importlib.import_module("shisa_de.readout")
                prompt = self.native.render(case.state, item, max_options=codes.MAX_CODES)
                inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
                slots = self.native.code_slots(len(question.options))
                self.settings["text_inference_policy"] = "released-SDK-DE2-direct-codebook-choice-transport"
            if inputs["input_ids"].shape[1] + 1 > self.context_limit:
                raise ValueError("Expanded Shisa input exceeds context limit; refusing truncation")
            if inputs["input_ids"][0, -1].item() != self.native.answer_prefix_id():
                raise ValueError("Shisa image input ends at an unexpected answer boundary")
            with self.torch.inference_mode():
                output = self.engine(**{key: value.to(self.device) for key, value in inputs.items()},
                                     logits_to_keep=1, use_cache=False)
                probabilities = output.logits[0, -1, [slot.token_id for slot in slots]].double().softmax(-1).cpu().tolist()
            keys = [option.id for option in question.options]
            distribution = dict(zip(keys, probabilities, strict=True))
            check_probabilities(distribution, [option.id for option in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id, probabilities=distribution))
        return predictions

    def close(self):
        if hasattr(self, "native"):
            self.native.close()
        super().close()
