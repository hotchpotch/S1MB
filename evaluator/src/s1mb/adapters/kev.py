"""Kev checkpoint loader and native strict input encoding."""

import importlib

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches


class KevAdapter(UpstreamAdapter):
    case_batch_size = 16

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
            "case_batch_size": self.case_batch_size,
            "microbatch_tokens": 4096,
            "renderer": "native-anonymous-choice-batched-v2",
        }

        self.enable_kernels()

    def predict(self, case):
        return self.predict_batch([case])[0]

    def predict_batch(self, cases):
        rows = []
        for index, case in enumerate(cases):
            for q in case.questions:
                req = self.api.SystemOneRequest(
                    state=case.state,
                    questions=questions_for_api([q], structured=True, anonymous_choice=True),
                )
                record, _ = self.api.to_record(req)
                enc = self.engine.encode(
                    self.tokenizer,
                    record,
                    max_state=self.context_limit,
                    max_branch=self.context_limit,
                    strict=True,
                )
                if len(enc["ids"]) > self.context_limit:
                    raise ValueError("Kev complete input exceeds context limit")
                rows.append((index, case.case_id, q, enc))
        rows.sort(key=lambda row: len(row[3]["ids"]))
        outputs = [{} for _ in cases]
        with self.torch.inference_mode():
            for batch in candidate_batches(rows, [len(r[3]["ids"]) for r in rows], 16, 4096):
                logits = self.engine.forward_batch([row[3] for row in batch])
                ps = [z[0].float().softmax(-1) for z in logits]
                values = self.torch.cat(ps).cpu().split([len(p) for p in ps])
                for (index, case_id, q, _), p in zip(batch, values, strict=True):
                    ids = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
                    outputs[index][q.id] = Prediction(
                        case_id=case_id,
                        question_id=q.id,
                        probabilities=dict(zip(ids, p.tolist(), strict=True)),
                    )
        return [
            [out[q.id] for q in case.questions] for out, case in zip(outputs, cases, strict=True)
        ]
