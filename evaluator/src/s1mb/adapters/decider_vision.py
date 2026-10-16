"""Decider's native vision readout with ordered multi-image and full-option transport."""

import importlib
import json

from s1mb.data import Prediction

from .upstream import UpstreamAdapter
from .vision_native import choice_transport, ordered_images, typed_questions


class DeciderVisionAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768):
        self.setup("decider-vision", model, revision, source, device)
        native = importlib.import_module("decider.vision.model")
        self.native = importlib.import_module("decider.infer")
        self.prompt = importlib.import_module("decider.prompt")
        self.no_shuffle = native._NoShuffle()
        self.image_marker = native.IMG
        self.engine = native.VisionDecisionModel(str(self.path), grad_ckpt=False).to(device).eval()
        self.attention_model = self.engine.lm
        self.set_attention("sdpa")
        self.enable_kernels()
        self.context_limit = context_limit
        self.image_receipts = {}
        self.settings.update({
            "dtype": "bfloat16", "max_input_tokens": context_limit,
            "input_length_policy": "reject-overflow", "max_candidates": 255,
            "media": "ordered-native-image-markers-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "renderer": "native-choice-numeric-score-full-options-v1",
            "capacity_extension": "native-processor-multiple-images-and-255-label-table",
            "calibration_policy": "native-vision-unscaled-letter-softmax",
        })

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        predictions = []
        for key, question in typed_questions(case).items():
            options = choice_transport(question)
            ex = self.native.Example(json.dumps(state, ensure_ascii=False), [
                self.native.Q(question["instructions"], options, 0)
            ])
            built = self.prompt.build(
                ex, self.engine.tok, self.no_shuffle, max_options=255, max_ctx_tokens=10**9
            )
            if built["nopts"] != [len(options)] or built["perms"] != [list(range(len(options)))]:
                raise ValueError("Native Decider changed candidate membership or order")
            prefix = "\n".join(f"{name}:\n{self.image_marker}" for name in case.images)
            text = prefix + self.engine.tok.decode(built["ids"])
            inputs = self.engine.proc(images=images or None, text=[text], return_tensors="pt")
            ids = inputs["input_ids"][0].tolist()
            if len(ids) > self.context_limit:
                raise ValueError("Decider visual input exceeds context limit")
            found = [i for i in range(2, len(ids))
                     if ids[i] == self.engine.slot_tok and ids[i - 1] == self.engine.colon
                     and self.engine.answer_tok in ids[max(0, i - 5):i]]
            if not found:
                raise ValueError("Decider answer slot is missing")
            inputs["slot_idx"] = self.torch.tensor([found[-1]])
            inputs["slot_batch"] = self.torch.tensor([0])
            inputs["nopts"] = self.torch.tensor([len(options)])
            with self.torch.inference_mode():
                probabilities = self.engine.slot_logits(inputs)[0, :len(options)].softmax(-1)
            q = next(q for q in case.questions if q.id == key)
            keys = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
            predictions.append(Prediction(
                case_id=case.case_id, question_id=key,
                probabilities=dict(zip(keys, probabilities.float().cpu().tolist(), strict=True)),
            ))
        return predictions
