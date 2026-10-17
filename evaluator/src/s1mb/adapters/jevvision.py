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
        self.text_adapter = None
        self.context_limit = context_limit
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
            "route": "released-native-text-and-visual-routes-v1", "max_images": 4,
            "text_checkpoint_subdirectory": "text/jevvision-text",
            "calibration_policy": "native-pointer-softmax",
        })

    def text_backend(self):
        """Load the bundle's declared default text adapter through its native backend."""
        if self.text_adapter is None:
            from .kev import KevAdapter

            self.text_adapter = KevAdapter(
                self.model_id, self.revision, str(self.path / "runtime"), self.device,
                context_limit=min(self.context_limit, 80000), case_batch_size=1,
                checkpoint_subdirectory="text/jevvision-text",
                loader_options={},
            )
            self.settings["active_route"] = "jevvision-text"
            self.settings["text_settings"] = self.text_adapter.settings
        return self.text_adapter

    def predict(self, case):
        if not case.images:
            return self.text_backend().predict(case)
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
