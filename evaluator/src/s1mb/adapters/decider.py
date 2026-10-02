"""Native Decider rendering and calibration with independent, bounded GPU rows."""

import importlib

from .base import decode_answers
from .firelex_jeff import decision_row
from .upstream import UpstreamAdapter, candidate_batches


class DeciderAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("decider", model, revision, source, device)
        self.native = importlib.import_module("decider.infer")
        self.engine = self.native.Decider(
            str(self.path), device=device, dtype=self.torch.bfloat16, use_graphs=False
        )
        self.attention_model = self.engine.m.lm
        self.assemble = importlib.import_module("decider.systemone").assemble
        self.settings = {
            "dtype": "bfloat16",
            "max_input_tokens": 32768,
            "independent": True,
            "cuda_graphs": False,
            "case_batch_size": self.case_batch_size,
            "microbatch_tokens": 4096,
            "renderer": "native-independent-anonymous-choice-numeric-score-batched-v3",
            "input_length_policy": "reject-overflow",
        }

        self.enable_kernels()

    def predict(self, case):
        return self.predict_batch([case])[0]

    def predict_batch(self, cases):
        if len(cases) > 1:
            return [self.predict_batch([case])[0] for case in cases]
        rows, refs = [], []
        layout = "schema_first" if self.engine.schema_first else "state_first"
        for index, case in enumerate(cases):
            for q in case.questions:
                request = {q.id: decision_row(case.state, q)["question"]}
                rqs, mapping, items = self.engine._system_one_items(
                    case.state, request, independent=True, max_state_tokens=10**9, layout=layout
                )
                if any(len(item["ids"]) > 32768 for item in items):
                    raise ValueError("Decider input exceeds context limit; refusing truncation")
                refs.append((index, q.id, rqs, mapping, len(items)))
                rows.extend(items)
        ordered = sorted(enumerate(rows), key=lambda row: len(row[1]["ids"]))
        probabilities = {}
        with self.torch.inference_mode():
            for group in candidate_batches(
                ordered, [((len(row[1]["ids"]) + 63) // 64) * 64 for row in ordered], 16, 4096
            ):
                items = [row[1] for row in group]
                batch = self.native.collate(items, self.engine.m.tok.pad_token_id)
                logits = self.engine.m.slot_logits(
                    *[
                        batch[k].to(self.device)
                        for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")
                    ]
                )
                temperatures = self.native.TT.for_items(self.engine.T, self.engine.T_by_type, items)
                p = (
                    self.native.TT.scaled_softmax(
                        logits, self.native.TT.slot_temperatures(temperatures, items)
                    )
                    .cpu()
                    .tolist()
                )
                offset = 0
                for i, item in group:
                    count = len(item["slots"])
                    probabilities[i] = p[offset : offset + count]
                    offset += count
        answers = [{} for _ in cases]
        offset = 0
        for index, _, rqs, mapping, count in refs:
            p = [
                v for row in [probabilities[i] for i in range(offset, offset + count)] for v in row
            ]
            offset += count
            answers[index].update(self.assemble(rqs, mapping, p))
        return [
            decode_answers(case, answer, rounded=True, anonymous_choice=True)
            for case, answer in zip(cases, answers, strict=True)
        ]
