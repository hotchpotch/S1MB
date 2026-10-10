"""Pinned Sifr option-key likelihoods with complete typed criteria and SDPA."""

import importlib
import json
import os

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, source_path


def sifr_questions(case):
    questions = questions_for_api(case.questions, structured=True, anonymous_choice=True)
    for question in case.questions:
        item = questions[question.id]
        # The native Noul renderer substitutes generic definitions, and its
        # Score primitive is unsupported. Explicit Choice criteria retain both.
        if question.task == "score":
            item["criteria"] = {
                f"option_{i}": {"value": option.value, "description": description}
                for i, (option, description) in enumerate(
                    zip(question.options, item["criteria"], strict=True)
                )
            }
        item["type"] = "choice"
    return questions


class SifrAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Sifr context_limit must be positive")
        if os.environ.get("SIFR_KV", "0") not in {"0", ""}:
            raise ValueError("Sifr requires the native full-sequence scoring path (SIFR_KV=0)")
        self.setup("sifr", model, revision, source, device)
        _, native_digest = source_path(str(self.path))
        self.native = importlib.import_module("sifr_engine")
        self.engine = self.native.SifrEngine(
            model=str(self.path), device=device, dtype="bfloat16",
            max_tokens=context_limit or 32768,
        )
        self.engine.BATCH = 1
        capacity = self.engine.model.config.get_text_config().max_position_embeddings
        if context_limit is not None and context_limit > capacity:
            raise ValueError("Sifr context limit exceeds checkpoint capacity")
        temperature = json.loads((self.path / "temperatures.json").read_text())["temperature"]
        if temperature != 1.0:
            raise ValueError("This Sifr native scorer requires its released identity temperature")
        self.attention_model = self.engine.model
        self.set_attention("sdpa")
        self.settings.update({
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.engine.limit,
            "checkpoint_position_limit": capacity,
            "input_length_policy": "reject-overflow-native",
            "branch_batch_size": 1,
            "scoring": "native-option-key-mean-log-likelihood-softmax",
            "temperature": temperature,
            "native_checkpoint_python_sha256": native_digest,
            "typed_mapping": "all-primitives-explicit-choice-criteria",
            "renderer": "native-structured-anonymous-numeric-score-v1",
        })

    def predict(self, case):
        questions = sifr_questions(case)
        predictions = []
        for question in case.questions:
            response, _ = self.engine(case.state, {"decision": questions[question.id]})
            answers = response["answers"]
            if set(answers) != {"decision"}:
                raise ValueError("Sifr returned incorrect answer count")
            probabilities = answers["decision"]["probabilities"]
            keys = (
                [option.id for option in question.options] if question.task == "noul"
                else [f"option_{i}" for i in range(len(question.options))]
            )
            check_probabilities(probabilities, keys)
            aligned = {
                option.id: probabilities[key]
                for option, key in zip(question.options, keys, strict=True)
            }
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=aligned
            ))
        return predictions
