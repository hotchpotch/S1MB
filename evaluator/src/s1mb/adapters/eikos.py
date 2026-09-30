"""Eikos native candidate-letter decisions on one or explicitly visible GPUs."""

import importlib

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, decoder_device_map


class EikosAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("eikos", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        self.native = importlib.import_module("letter_adapter")
        self.engine = self.native.LetterAdapter(
            str(self.path), device=device, max_tokens=32768, temp=1.0, readout="letter"
        )
        tok = transformers.AutoTokenizer.from_pretrained(str(self.path))
        count = self.torch.cuda.device_count()
        placement = decoder_device_map(self.path, device, self.torch, transformers)
        backbone = transformers.AutoModelForCausalLM.from_pretrained(
            str(self.path),
            dtype=self.torch.bfloat16,
            attn_implementation="sdpa",
            device_map=placement,
        ).eval()
        if any(p.device.type != "cuda" for p in backbone.parameters()):
            raise RuntimeError("Eikos must fit entirely on the explicitly visible GPUs")
        self.engine._loaded = (backbone, tok, self.engine._letter_ids(tok))
        self.attention_model = backbone
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": 32768,
            "case_batch_size": 1,
            "temperature": 1.0,
            "gpu_count": count,
            "device_map": placement,
            "renderer": "native-semif-anonymous-choice-v1",
            "extended_input_condition": "32768 tokens instead of runtime 12000 default",
        }
        if self.native.STYLE != "semif":
            raise ValueError("Eikos requires its released semif prompt")
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for q in case.questions:
            question = questions_for_api([q], anonymous_choice=True)[q.id]
            options = self.native.options_of(question)
            with self.torch.inference_mode():
                raw, _ = self.engine.dist(case.state, question, options)
            keys = (
                ["yes" if o.id == "true" else "no" for o in q.options]
                if q.task == "noul"
                else [str(i) if q.task == "score" else f"option_{i}" for i in range(len(q.options))]
            )
            if set(raw) != set(keys):
                raise ValueError("Eikos returned unexpected candidates")
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: raw[k] for o, k in zip(q.options, keys, strict=True)},
                )
            )
        return predictions
