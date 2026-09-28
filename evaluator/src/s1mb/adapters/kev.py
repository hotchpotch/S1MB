"""Kev checkpoint loader and native strict input encoding."""

import importlib

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter


class KevAdapter(UpstreamAdapter):
    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("kev", model, revision, source, device)
        cp = importlib.import_module("kev.checkpoint")
        self.api = importlib.import_module("kev.api")
        checkpoint = cp.Checkpoint(str(self.path))
        self.tokenizer, self.engine = checkpoint.load(
            device,
            cp.LoadOptions(dtype=self.torch.bfloat16, attn="sdpa", cuda_graphs=False, fused=False),
        )
        self.engine.eval()
        self.attention_model = self.engine.lm
        self.context_limit = context_limit or 8192
        self.settings = {
            "dtype": "bfloat16",
            "temperature": checkpoint.meta.temperature,
            "max_state": self.context_limit,
            "max_branch": self.context_limit,
            "input_policy": "strict native encoding",
        }

    def predict(self, case):
        predictions = []
        for q in case.questions:
            req = self.api.SystemOneRequest(state=case.state, questions=questions_for_api([q]))
            record, meta = self.api.to_record(req)
            enc = self.engine.encode(
                self.tokenizer,
                record,
                max_state=self.context_limit,
                max_branch=self.context_limit,
                strict=True,
            )
            with self.torch.inference_mode():
                probs = self.engine.forward(enc)[0].float().softmax(-1).cpu().tolist()
            ids = meta[0]["keys"] if q.task != "score" else [o.id for o in q.options]
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities=dict(zip(ids, probs, strict=True)),
                )
            )
        return predictions
