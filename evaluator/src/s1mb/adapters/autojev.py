"""AutoJev's published native decision backbone, readout and temperature."""

import importlib
import json
import sys

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import FirelexJeffAdapter
from .sifr import sifr_questions


class AutoJevAdapter(FirelexJeffAdapter):
    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("AutoJev context_limit must be positive")
        if sys.version_info < (3, 12):
            raise RuntimeError("AutoJev's upstream runtime requires Python 3.12 or later")
        self.setup("autojev", model, revision, source, device)
        config = json.loads((self.path / "decision_config.json").read_text())
        self.native = importlib.import_module("autojev.model")
        self.engine = self.native.DecisionModel(checkpoint=self.path, device=device).eval()
        self.attention_model = self.engine.backbone
        self.max_candidates = min(self.native.MAX_OPTIONS, len(self.engine.codes))
        self.context_limit = context_limit or 8192
        if self.context_limit > self.attention_model.config.get_text_config().max_position_embeddings:
            raise ValueError("AutoJev context limit exceeds checkpoint capacity")
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "temperature": self.engine.temperature,
            "max_input_tokens": self.context_limit,
            "max_candidates": self.max_candidates,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "base_model": config["base_model"],
            "base_revision": config["revision"],
            "renderer": "native-anonymous-choice-numeric-score-v1",
            "attention_mode": self.engine.attention_mode,
            "pooling": self.engine.pooling,
            "typed_mapping": "all-choice-preserving-authored-order-and-structured-numeric-score",
        }
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        rendered = sifr_questions(case)
        for question in case.questions:
            batch = self.engine.prepare(
                [{"state": case.state, "question": rendered[question.id]}],
                max_length=self.context_limit,
            )
            with self.torch.inference_mode():
                values = (self.engine(batch) / self.engine.temperature).softmax(-1)
                values = values[0, :len(question.options)].float().cpu().tolist()
            keys = [option.id for option in question.options]
            probabilities = dict(zip(keys, values, strict=True))
            check_probabilities(probabilities, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=probabilities,
            ))
        return predictions
