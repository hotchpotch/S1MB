"""JevK5's native calibrated letter-logit readout."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter, candidate_batches


class JevK5Adapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("jevk5", model, revision, source, device)
        self.engine = importlib.import_module("jevk5.runtime").JevK5(
            str(self.path), device=device, dtype=self.torch.bfloat16, graphs=False
        )
        self.attention_model = self.engine.model
        native_encode = self.engine.encode

        def strict_encode(*args, **kwargs):
            ids = native_encode(*args, **kwargs)
            if len(ids) > 32768:
                raise ValueError("JevK5 input exceeds context limit; refusing truncation")
            return ids

        self.engine.encode = strict_encode
        self.native = importlib.import_module("jevk5.runtime")
        self.settings = {
            "dtype": "bfloat16",
            "temperature": self.engine.temperature,
            "large_candidate_policy": "native knockout above 16 candidates",
            "cuda_graphs": False,
            "max_input_tokens": 32768,
            "case_batch_size": self.case_batch_size,
            "microbatch_tokens": 4096,
            "renderer": "native-letter-logits-anonymous-choice-batched-v2",
        }

        self.enable_kernels()

    def predict(self, case):
        return self.predict_batch([case])[0]

    def predict_batch(self, cases):
        if len(cases) > 1:
            return [self.predict(case) for case in cases]
        rows = []
        answers = [{} for _ in cases]
        for index, case in enumerate(cases):
            for q in case.questions:
                definition = questions_for_api([q], anonymous_choice=True)[q.id]
                options = self.native.decision_options(definition)
                if len(options) > 16:
                    answers[index][q.id] = self.engine.decide(case.state, definition)
                    continue
                ids = self.engine.encode(
                    case.state, definition["instructions"], [v for _, v in options]
                )
                rows.append((index, q.id, definition, options, ids))
        rows.sort(key=lambda row: len(row[4]))
        with self.torch.inference_mode():
            for group in candidate_batches(rows, [len(row[4]) for row in rows], 16, 4096):
                width = max(len(row[4]) for row in group)
                ids = self.torch.zeros(
                    (len(group), width), dtype=self.torch.long, device=self.device
                )
                for i, row in enumerate(group):
                    ids[i, : len(row[4])] = self.torch.tensor(row[4], device=self.device)
                last = self.torch.tensor([len(row[4]) - 1 for row in group], device=self.device)
                logits = self.engine._slot_logits(ids, last).float() / self.engine.temperature
                ps = [logits[i, : len(row[3])].softmax(-1) for i, row in enumerate(group)]
                values = self.torch.cat(ps).cpu().split([len(p) for p in ps])
                for (index, qid, definition, options, tokens), p in zip(group, values, strict=True):
                    answers[index][qid] = self.native.answer(
                        definition,
                        dict(zip([k for k, _ in options], p.tolist(), strict=True)),
                        len(tokens),
                    )
        return [
            decode_answers(case, answer, rounded=True, anonymous_choice=True)
            for case, answer in zip(cases, answers, strict=True)
        ]
