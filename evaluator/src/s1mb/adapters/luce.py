"""Published Luce/Ouro checkpoint's standalone, native scoring implementation."""

import importlib.util
import sys

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches


class LuceAdapter(UpstreamAdapter):
    def __init__(self, model, revision, source, device):
        self.setup("luce", model, revision, source, device)
        # The published Ouro checkpoint differs from the generic Luce loader.
        spec = importlib.util.spec_from_file_location(
            "s1mb_luce_checkpoint", self.path / "inference.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError("The published checkpoint must include inference.py")
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)
        self.engine = self.module.DecisionModel(
            str(self.path), device=device, dtype=self.torch.bfloat16
        )
        self.attention_model = self.engine.body
        self.settings = {
            "dtype": "bfloat16",
            "config": self.engine.cfg,
            "base_revision": getattr(self.engine.body.config, "_commit_hash", None),
            "max_query_len": self.engine.max_query_len,
            "candidate_batch_size": 8,
            "candidate_batch_token_budget": 8192,
            "input_policy": "native suffix retention; explicit dataset Noul descriptions",
        }

    def predict(self, case):
        out = []
        for q in case.questions:
            api = questions_for_api([q])[q.id]
            native = {"type": q.task, "prompt": api["instructions"]}
            if q.task == "choice":
                native["options"] = api["criteria"]
            elif q.task == "score":
                native["levels"] = api["criteria"]
            item = self.module.build_item(self.module.state_to_text(case.state), native)
            if q.task == "noul":
                opts = {o.id: o.description for o in q.options}
                item.descriptions = [opts["true"], opts["false"]]
            # Candidate paths are independent. Concatenate logits before one softmax.
            pairs = self.engine._pairs(item)
            lengths = [sum(len(ids) for ids in self.engine._ids(*pair)) for pair in pairs]
            scores = self.torch.cat(
                [
                    self.engine._score_pairs(batch)
                    for batch in candidate_batches(pairs, lengths, token_budget=8192)
                ]
            )
            temperature = self.engine.temperatures.get(q.task, self.engine.temperature)
            probs = (scores.float() / temperature).softmax(-1).cpu().tolist()
            ids = ["true", "false"] if q.task == "noul" else [o.id for o in q.options]
            out.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities=dict(zip(ids, probs, strict=True)),
                )
            )
        return out
