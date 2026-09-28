"""Bridge to JevForge's native predictor and checkpoint calibration."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter, candidate_batches, state_text


class JevForgeAdapter(UpstreamAdapter):
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
            grouped = []
            for example in examples:
                leaves = example["leaves"]
                chunks = [
                    native_forward([{**example, "leaves": batch}])[0]
                    for batch in candidate_batches(leaves, [len(leaf) for leaf in leaves])
                ]
                grouped.append(self.torch.cat(chunks))
            return grouped

        self.engine._forward = bounded_forward
        self.settings = {
            "dtype": "bfloat16",
            "candidate_batch_size": 8,
            "candidate_batch_token_budget": 4096,
            "checkpoint_max_length": checkpoint_limit,
            "temperature": self.engine.temperature,
            "max_length": self.engine.max_length,
            "input_policy": "native strict encoder; over-limit input fails",
        }

    def predict(self, case):
        answers = {}
        for q in case.questions:
            answers.update(self.engine.decide(state_text(case.state), questions_for_api([q])))
        return decode_answers(case, answers)
