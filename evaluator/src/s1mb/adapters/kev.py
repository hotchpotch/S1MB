"""Kev checkpoint loader and native strict input encoding."""

import importlib

from s1mb.data import Prediction

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, candidate_batches


class KevAdapter(UpstreamAdapter):
    case_batch_size = 16

    def __init__(self, model, revision, source, device, context_limit=None, case_batch_size=16):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Kev context_limit must be positive")
        if case_batch_size < 1:
            raise ValueError("case_batch_size must be positive")
        self.case_batch_size = case_batch_size
        self.setup("kev", model, revision, source, device)
        cp = importlib.import_module("kev.checkpoint")
        self.api = importlib.import_module("kev.api")
        checkpoint = cp.Checkpoint(str(self.path))
        if not checkpoint.meta.base_revision:
            hub = importlib.import_module("huggingface_hub")
            checkpoint.meta.base_revision = hub.model_info(checkpoint.meta.base).sha
            if not checkpoint.meta.base_revision:
                raise ValueError("Hub did not resolve Kev's base checkpoint revision")
        self.tokenizer, self.engine = checkpoint.load(
            device,
            cp.LoadOptions(dtype=self.torch.bfloat16, attn="sdpa", cuda_graphs=False, fused=False),
        )
        self.engine.eval()
        self.attention_model = self.engine.lm
        self.context_limit = context_limit or 8192
        if self.context_limit > self.attention_model.config.get_text_config().max_position_embeddings:
            raise ValueError("Kev context limit exceeds checkpoint capacity")
        self.settings = {
            "dtype": "bfloat16",
            "temperature": checkpoint.meta.temperature,
            "base_model": checkpoint.meta.base,
            "base_revision": checkpoint.meta.base_revision,
            "max_state": self.context_limit,
            "max_branch": self.context_limit,
            "input_policy": "strict native encoding",
            "case_batch_size": self.case_batch_size,
            "microbatch_tokens": 4096,
            "renderer": "native-anonymous-choice-numeric-score-batched-v3",
            "typed_mapping": "all-choice-preserving-authored-order-and-structured-numeric-score",
        }

        self.enable_kernels()

    def predict(self, case):
        return self.predict_batch([case])[0]

    def predict_batch(self, cases):
        if len(cases) > self.case_batch_size:
            return [
                output
                for offset in range(0, len(cases), self.case_batch_size)
                for output in self.predict_batch(cases[offset : offset + self.case_batch_size])
            ]
        rows = []
        for index, case in enumerate(cases):
            questions = sifr_questions(case)
            for q in case.questions:
                req = self.api.SystemOneRequest(
                    state=case.state,
                    questions={"decision": questions[q.id]},
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
                    ids = [o.id for o in q.options]
                    outputs[index][q.id] = Prediction(
                        case_id=case_id,
                        question_id=q.id,
                        probabilities=dict(zip(ids, p.tolist(), strict=True)),
                    )
        return [
            [out[q.id] for q in case.questions] for out, case in zip(outputs, cases, strict=True)
        ]
