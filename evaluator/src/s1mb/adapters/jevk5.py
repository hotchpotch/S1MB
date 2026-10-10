"""JevK5's native calibrated letter-logit readout."""

import importlib

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


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
            "knockout_temperature": self.engine.knockout_temperature,
            "cuda_graphs": False,
            "max_input_tokens": 32768,
            "case_batch_size": self.case_batch_size,
            "renderer": "native-structured-choice-authored-noul-numeric-score",
        }

        self.set_attention("sdpa")

    def predict(self, case):
        wire = sifr_questions(case)
        predictions = []
        for question in case.questions:
            definition = wire[question.id]
            probabilities, _ = self.engine.probabilities(case.state, definition)
            keys = list(definition["criteria"])
            check_probabilities(probabilities, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={o.id: probabilities[key] for o, key in
                               zip(question.options, keys, strict=True)},
            ))
        return predictions
