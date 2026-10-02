"""Hopper's native JSON prompt, LoRA, restricted readout and calibration."""

import importlib
import json
import string
from typing import Any, cast

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .jevlite import extended_labels
from .upstream import UpstreamAdapter, state_text


class HopperAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("hopper", model, revision, source, device)
        native = importlib.import_module("hopper_decisions.model")
        self.prompt = cast(Any, importlib.import_module("hopper_decisions.prompt"))
        self.calibration = importlib.import_module("hopper_decisions.calibration")
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            native.BASE, revision=native.REVISION
        )
        self.codes = extended_labels(tokenizer, list(string.ascii_uppercase), max_candidates or 26)
        self.prompt.LETTERS = self.codes
        calibration_path = self.path / "hopper.json"
        if not calibration_path.exists():
            calibration_path = self.source / "hopper_decisions/maps/hopper.json"
        self.engine = native.Decider(
            adapter=str(self.path),
            calibration_map=calibration_path,
            device=device,
            attention="sdpa",
            prefix_cache=False,
            allow_slow_kernels=True,
            warm_lengths=(),
            cuda_graphs=False,
            shortlist=None,
        )
        self.attention_model = self.engine.model
        self.context_limit = context_limit or 8192
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "base_model": native.BASE,
            "base_revision": native.REVISION,
            "max_input_tokens": self.context_limit,
            "max_candidates": len(self.codes),
            "native_max_candidates": 26,
            "answer_codes": self.codes,
            "calibration": json.loads(calibration_path.read_text()),
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "shortlist": None,
            "renderer": "native-json-authored-noul-numeric-score-v1",
        }
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for question in case.questions:
            if len(question.options) > len(self.codes):
                raise ValueError("Hopper candidate capacity exceeded; refusing to drop options")
            item = questions_for_api([question], structured=True, anonymous_choice=True)[
                question.id
            ]
            shown = []
            for index, option in enumerate(question.options):
                key = (
                    option.id
                    if question.task == "noul"
                    else str(option.value)
                    if question.task == "score"
                    else f"option_{index}"
                )
                description = state_text(
                    json.loads(option.description_json)
                    if option.description_json is not None
                    else option.description
                )
                shown.append((f"option_{index}", f"{key}: {description}"))
            example = {"kind": question.task, "document": state_text(case.state), "policy": ""}
            messages = self.prompt.messages(example, state_text(item["instructions"]), shown)
            ids = self.prompt.chat_ids(self.engine.tokenizer, messages)
            if len(ids) > self.context_limit:
                raise ValueError("Hopper input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                values = self.engine.letter_probs(ids, len(shown))
            raw = dict(zip([o.id for o in question.options], values, strict=True))
            probabilities = self.calibration.apply(self.engine.map, example, raw)
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
