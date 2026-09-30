"""Jebadiah's native AINode renderer and calibrated FP32 candidate readout."""

import importlib
import json

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter


class JebadiahAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("jebadiah", model, revision, source, device)
        self.native = importlib.import_module("jebadiah_model")
        adapter_config = self.path / "adapter_config.json"
        if adapter_config.exists():
            base = json.loads(adapter_config.read_text())["base_model_name_or_path"]
            base_revision = "68c46c4b3498877f3ef123c856ecfde50c39f404"
            load_path, load_revision = base, base_revision
            checkpoint_format = "lora-adapter"
        else:
            manifest = json.loads((self.path / "jebadiah.json").read_text())
            base, base_revision = manifest["base"].rsplit("@", 1)
            load_path, load_revision = str(self.path), None
            checkpoint_format = "merged-bfloat16"
        tokenizer = self.native.load_tokenizer(load_path, load_revision)
        model_object = self.native.load_base(
            load_path,
            load_revision,
            attn_implementation="sdpa",
            dtype=self.torch.bfloat16,
            device=device,
        )
        if checkpoint_format == "lora-adapter":
            model_object = self.native.load_adapter(model_object, str(self.path))
        self.engine = self.native.Scorer(
            model_object,
            tokenizer,
            max_tokens=32768,
            temperatures=self.native.read_temperatures(str(self.path)),
            device=device,
        )
        self.attention_model = model_object
        self.settings = {
            "dtype": "bfloat16-backbone-float32-candidate-logits",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": 32768,
            "case_batch_size": 1,
            "base_model": base,
            "base_revision": base_revision,
            "checkpoint_format": checkpoint_format,
            "temperatures": self.engine.temperatures,
            "renderer": "native-ainode-extended-alphabet-anonymous-choice-v1",
            "extended_input_condition": "32768 tokens instead of runtime 2048 default",
        }
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for q in case.questions:
            native = questions_for_api([q], anonymous_choice=True)[q.id]
            rendered = self.engine.render(case.state, native)
            if rendered.truncated:
                raise ValueError("Jebadiah input exceeds context limit; refusing truncation")
            probabilities = self.engine.score_rendered([(rendered, q.task)])[0]
            raw = dict(zip(rendered.keys, probabilities, strict=True))
            keys = [
                o.id if q.task == "noul" else str(i) if q.task == "score" else f"option_{i}"
                for i, o in enumerate(q.options)
            ]
            if set(raw) != set(keys):
                raise ValueError("Jebadiah returned unexpected candidates")
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: raw[k] for o, k in zip(q.options, keys, strict=True)},
                )
            )
        return predictions
