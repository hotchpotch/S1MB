"""Native AJev label-logit calibration with complete input admission."""

import importlib
import json
import math

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, checkpoint_path


def ajev_decisions(native, request, case):
    """Use native typed hints and temperatures while retaining authored criteria order."""
    wire = sifr_questions(case)
    decisions = []
    for question in case.questions:
        questions = {"decision": wire[question.id]}
        decision = native.decisions_from_jev(
            case.state, request.plain_questions(questions),
            lang=request.detect_lang(case.state, questions),
        )[0]
        decision.type = question.task
        decision.validate()
        decisions.append(decision)
    return decisions


class AJevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, base_revision, context_limit=None):
        if not base_revision:
            raise ValueError("AJev requires an explicit base revision")
        self.setup("ajev", model, revision, source, device)
        self.native = importlib.import_module("ajev.schema")
        self.request = importlib.import_module("ajev.jev_request")
        self.predictor = importlib.import_module("ajev.lm.predictor")
        config = json.loads((self.path / "ajev_lm_config.json").read_text())
        base_path, resolved = checkpoint_path(config["base_model"], base_revision)
        self.engine = self.predictor.LMPredictor(
            str(base_path), adapter=str(self.path), device=device,
            max_state_tokens=10 ** 9, batch_tokens=24000, prefix_cache=False,
            dtype=self.torch.bfloat16, merge=False,
        )
        self.attention_model = self.engine.model
        self.set_attention("sdpa")
        capacity = self.engine.model.config.get_text_config().max_position_embeddings
        self.limit = 32768 if context_limit is None else context_limit
        if not 1 <= self.limit <= capacity:
            raise ValueError("AJev context limit exceeds checkpoint capacity")
        if any(not math.isfinite(t) or t <= 0 for t in self.engine.temperatures.values()):
            raise ValueError("Invalid released AJev calibration temperature")
        self.settings.update({
            "dtype": "bfloat16", "base_model": config["base_model"], "base_revision": resolved,
            "temperatures": self.engine.temperatures, "attention_implementation": "sdpa",
            "max_input_tokens": self.limit, "checkpoint_position_limit": capacity,
            "input_length_policy": "reject-overflow-with-native-state-truncation-disabled",
            "prefix_cache": False, "lora_merge": False, "batch_tokens": 24000,
            "renderer": "native-typed-hints-authored-noul-numeric-score",
            "readout": "native-bare-and-space-label-logsumexp-per-type-softmax",
        })

    def predict(self, case):
        decisions = ajev_decisions(self.native, self.request, case)
        for decision in decisions:
            ids = self.predictor.prompt_ids(self.engine.tok, decision, 10 ** 9)
            if len(ids) > self.limit:
                raise ValueError("Complete AJev prompt exceeds context limit")
        values = self.engine.predict(decisions)
        predictions = []
        for question, row in zip(case.questions, values, strict=True):
            probabilities = dict(zip([o.id for o in question.options], row, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions
