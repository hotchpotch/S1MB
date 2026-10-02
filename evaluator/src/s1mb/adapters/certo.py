"""Certo's native independent-option scorer without tokenizer truncation."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, checkpoint_path, state_text


def certo_inputs(state, question):
    item = questions_for_api([question], structured=True, anonymous_choice=True)[question.id]
    descriptions = item["criteria"] if question.task == "score" else list(item["criteria"].values())
    if question.task == "score":
        descriptions = [
            {"value": option.value, "description": description}
            for option, description in zip(question.options, descriptions, strict=True)
        ]
    return (
        state_text(item["instructions"]) + "\n\n" + state_text(state),
        [state_text(description) for description in descriptions],
    )


class CertoAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("certo", model, revision, source, device)
        cfg = json.loads((self.path / "certo_config.json").read_text())
        self.context_limit = context_limit or 8192
        base, base_revision = checkpoint_path(cfg["backbone"], "main")
        self.native = importlib.import_module("model_generic")
        self.engine = self.native.GenericDecisionModel(str(base), heads=cfg.get("heads", 8))
        self.engine.load_state_dict(
            self.torch.load(
                self.path / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        self.engine.to(device).eval()
        self.attention_model = self.engine.bert
        self.tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
            str(self.path)
        )
        self.temperature = float(cfg.get("temperature", 1.0))
        if not 0 < self.temperature < float("inf"):
            raise ValueError("Invalid Certo calibration temperature")
        self.settings = {
            "dtype": "float32",
            "max_input_tokens": self.context_limit,
            "input_length_policy": "reject-overflow",
            "native_state_limit": 64,
            "native_option_limit": 48,
            "extended_input_condition": "Dynamic full inputs instead of native 64/48 truncation",
            "temperature": self.temperature,
            "base_model": cfg["backbone"],
            "base_revision": base_revision,
            "case_batch_size": 1,
            "candidate_batch_size": 8,
            "renderer": "native-independent-options-numeric-score-v1",
            "probabilities": "native-softmax-before-display-rounding",
        }
        self.set_attention("sdpa")

    def predict(self, case):
        predictions = []
        for q in case.questions:
            state, options = certo_inputs(case.state, q)
            encoded = self.tokenizer([state, *options], truncation=False)
            if any(len(ids) > self.context_limit for ids in encoded["input_ids"]):
                raise ValueError("Certo input exceeds context limit; refusing truncation")
            se = self.tokenizer([state], return_tensors="pt", truncation=False).to(self.device)
            logits = []
            with self.torch.inference_mode():
                for start in range(0, len(options), 8):
                    chunk = options[start : start + 8]
                    oe = self.tokenizer(
                        chunk,
                        padding=True,
                        return_tensors="pt",
                        truncation=False,
                    ).to(self.device)
                    logits.append(
                        self.engine(
                            se["input_ids"],
                            se["attention_mask"],
                            oe["input_ids"].unsqueeze(0),
                            oe["attention_mask"].unsqueeze(0),
                            self.torch.ones(
                                1, len(chunk), dtype=self.torch.bool, device=self.device
                            ),
                        )[0]
                    )
                values = (
                    (self.torch.cat(logits).float() / self.temperature).softmax(-1).cpu().tolist()
                )
            probabilities = dict(zip([o.id for o in q.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in q.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities=probabilities,
                )
            )
        return predictions
