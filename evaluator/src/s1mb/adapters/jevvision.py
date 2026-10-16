"""JEVision's released visual sidecar and native pointer readout."""

import importlib
import json
import sys

from s1mb.data import Prediction

from .upstream import UpstreamAdapter
from .vision_native import choice_transport, ordered_images, typed_questions


class JEVisionAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768):
        self.setup("jevvision", model, revision, source, device)
        sys.path.insert(0, str(self.path / "runtime"))
        native = importlib.import_module("kev.visual_model")
        self.engine = native.VisualDecisionModel.from_run(
            self.path, device=device, max_input_tokens=context_limit
        )
        self.attention_model = self.engine.model
        self.set_attention("sdpa")
        self.enable_kernels()
        self.image_receipts = {}
        report = json.loads((self.path / "report.json").read_text())
        self.settings.update({
            "dtype": "float16-float32-head", "max_input_tokens": context_limit,
            "input_length_policy": "reject-overflow-native", "media": "native-ordered-pil-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "renderer": "native-pointer-authored-noul-numeric-score-v1",
            "base_model": report["model"], "base_revision": report["revision"],
            "route": "released-visual-sidecar-only", "max_images": 4,
            "calibration_policy": "native-pointer-softmax",
        })

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        self.image_receipts[case.case_id] = receipts
        predictions = []
        for key, question in typed_questions(case).items():
            record = {"state": json.dumps(state, ensure_ascii=False), "questions": [{
                "instr": question["instructions"], "options": choice_transport(question),
            }]}
            values = self.engine.probs(record, images)
            if len(values) != 1:
                raise ValueError("JEVision returned incorrect answer count")
            q = next(q for q in case.questions if q.id == key)
            keys = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
            predictions.append(Prediction(
                case_id=case.case_id, question_id=key,
                probabilities=dict(zip(keys, values[0].float().cpu().tolist(), strict=True)),
            ))
        return predictions
