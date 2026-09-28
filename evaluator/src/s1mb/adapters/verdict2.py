"""Original Verdict 2.0 marker model; requires its actual trained checkpoint."""

import importlib

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, state_text


class Verdict2Adapter(UpstreamAdapter):
    def __init__(self, model, revision, source, device):
        self.setup("verdict2", model, revision, source, device)
        checkpoint = self.path / "model.pt" if self.path.is_dir() else self.path
        self.module = importlib.import_module("verdict2.model")
        self.data = importlib.import_module("verdict2.data")
        state = self.torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
            state["backbone"]
        )
        self.engine = self.module.VerdictModel(state["backbone"])
        self.engine.load_state_dict(state["state_dict"], strict=True)
        self.engine.to(device).eval()
        self.settings = {
            "dtype": "float32",
            "max_len": 2048,
            "head_max_len": 1024,
            "temperature": self.engine.temperature.tolist(),
            "input_policy": "native prefix truncation; 2048/1024 explicit S1MB context profile",
        }

    def predict(self, case):
        output = []
        for q in case.questions:
            item = self.data.build_item(
                self.tokenizer,
                {"id": case.case_id, "state": state_text(case.state)},
                q.id,
                questions_for_api([q])[q.id],
                {},
                max_len=2048,
                head_max_len=1024,
            )
            if item is None:
                raise ValueError("Verdict candidate markers do not fit")
            tensor = lambda v: self.torch.tensor([v], device=self.device)
            batch = {
                "input_ids": tensor(item.ids),
                "attention_mask": tensor([1] * len(item.ids)),
                "qtype": self.torch.tensor([item.qtype], device=self.device),
                "marker_pos": tensor(item.markers),
                "marker_mask": tensor([True] * len(item.markers)),
            }
            with self.torch.inference_mode():
                logits = self.engine.option_logits(batch)
                scaled = self.module.apply_temperature(
                    logits, batch["qtype"], batch["marker_mask"].sum(-1), self.engine.temperature
                )
                p = scaled.float().softmax(-1)[0].cpu().tolist()
            ids = list(item.option_keys) if q.task != "score" else [o.id for o in q.options]
            output.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities=dict(zip(ids, p, strict=True)),
                )
            )
        return output
