"""Bridge to JevForge's native predictor and checkpoint calibration."""

import importlib

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches, state_text


class JevForgeAdapter(UpstreamAdapter):
    case_batch_size = 16

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("jevforge", model, revision, source, device)
        self.engine = importlib.import_module("jevforge.predict").Predictor(
            str(self.path), device_name=device, precision="bf16"
        )
        self.attention_model = self.engine.model.backbone
        checkpoint_limit = self.engine.max_length
        if context_limit is not None:
            self.engine.max_length = context_limit
        native_forward = self.engine._forward

        def bounded_forward(examples):
            leaves = [leaf for example in examples for leaf in example["leaves"]]
            ordered = sorted(enumerate(leaves), key=lambda row: len(row[1]))
            values = {}
            for batch in candidate_batches(ordered, [len(r[1]) for r in ordered], 32, 8192):
                scores = native_forward([{"leaves": [row[1] for row in batch]}])[0]
                for (index, _), score in zip(batch, scores, strict=True):
                    values[index] = score
            return self.torch.stack([values[i] for i in range(len(leaves))]).split(
                [len(e["leaves"]) for e in examples]
            )

        self.engine._forward = bounded_forward
        self.settings = {
            "dtype": "bfloat16",
            "candidate_batch_size": 32,
            "candidate_batch_token_budget": 8192,
            "case_batch_size": self.case_batch_size,
            "renderer": "native-anonymous-choice-batched-v2",
            "checkpoint_max_length": checkpoint_limit,
            "temperature": self.engine.temperature,
            "max_length": self.engine.max_length,
            "input_policy": "native strict encoder; over-limit input fails",
        }

        self.enable_kernels()

    def predict(self, case):
        return self.predict_batch([case])[0]

    def predict_batch(self, cases):
        encode = importlib.import_module("jevforge.encode").encode_examples
        examples, refs = [], []
        for index, case in enumerate(cases):
            examples.extend(
                encode(
                    {
                        "state": state_text(case.state),
                        "questions": questions_for_api(case.questions, anonymous_choice=True),
                    },
                    self.engine.tokenizer,
                    self.engine.max_length,
                )
            )
            refs.extend((index, case.case_id, q) for q in case.questions)
        outputs = [[] for _ in cases]
        for (index, case_id, q), logits in zip(refs, self.engine._forward(examples), strict=True):
            p = (logits.float() / self.engine.temperature_for(q.task)).softmax(-1).cpu().tolist()
            ids = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
            outputs[index].append(
                Prediction(
                    case_id=case_id, question_id=q.id, probabilities=dict(zip(ids, p, strict=True))
                )
            )
        return outputs
