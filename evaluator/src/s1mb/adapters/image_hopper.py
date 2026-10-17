"""Image Hopper's released image routes, LoRA and calibrated choice readout."""

import importlib

from .base import decode_answers
from .upstream import UpstreamAdapter
from .vision_native import choice_transport, ordered_images, typed_questions


def public_hopper_provenance(value):
    """Retain native hashes and settings while omitting machine-local paths."""
    if isinstance(value, dict):
        return {key: public_hopper_provenance(item) for key, item in value.items() if key != "path"}
    if isinstance(value, list):
        return [public_hopper_provenance(item) for item in value]
    return value


class ImageHopperAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768):
        if context_limit < 1:
            raise ValueError("Image Hopper context_limit must be positive")
        self.setup("image-hopper", model, revision, source, device)
        native = importlib.import_module("open_decisions.image_jev.release.predictor")
        self.engine, provenance = native.load_predictor(
            calibration_path=self.path / "calibration.json", model_name=model,
            adapter_path=self.path / "adapter", base_hashes_path=self.path / "base-model-files.json",
            device=device, max_input_tokens=context_limit, allow_slow_path=True,
            serving_options="lean_lora,uint8_pixels,device_readout,image_token_cap_default=768",
        )
        # The exact lean-LoRA serving path holds trained factors as buffers.
        # Count storage-sharing parameter views without changing native inference.
        factors = [value for name, value in self.engine.model.named_buffers()
                   if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}]
        self.engine.lora_parameters = self.torch.nn.ParameterList([
            self.torch.nn.Parameter(value, requires_grad=False) for value in factors
        ])
        self.attention_model = self.engine.model
        self.image_receipts = {}
        self.settings.update({
            "dtype": "bfloat16", "native_provenance": public_hopper_provenance(provenance),
            "max_input_tokens": context_limit, "input_length_policy": "reject-overflow-native",
            "max_images": 4, "max_candidates": 26,
            "renderer": "native-choice-authored-noul-numeric-score-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
        })

    def predict(self, case):
        import base64
        import io

        images, state, receipts = ordered_images(case)
        if not images:
            raise ValueError("Image Hopper requires at least one image; text scopes unsupported")
        self.image_receipts[case.case_id] = receipts
        encoded = []
        for image in images:
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            encoded.append("data:image/png;base64," + base64.b64encode(stream.getvalue()).decode())
        answers = {}
        for key, question in typed_questions(case).items():
            descriptions = choice_transport(question)
            if len(descriptions) > 26:
                raise ValueError("Image Hopper supports at most 26 options")
            criteria = {f"option_{i}": value for i, value in enumerate(descriptions)}
            response = self.engine({"images": encoded, "state": state, "questions": {
                "decision": {"type": "choice", "instructions": question["instructions"],
                             "criteria": criteria},
            }})
            answer = response["answers"]["decision"]
            if question["type"] == "noul":
                answer = {"type": "noul", "noul": answer["probabilities"]["option_1"]}
            elif question["type"] == "score":
                answer = {"type": "score", "probabilities": {
                    str(i): answer["probabilities"][f"option_{i}"]
                    for i, option in enumerate(next(q for q in case.questions if q.id == key).options)
                }}
            answers[key] = answer
        return decode_answers(case, answers, anonymous_choice=True)
