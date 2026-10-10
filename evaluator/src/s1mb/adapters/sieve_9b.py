"""Sieve-9B's native causal-row pointer head and released calibration."""

import importlib
import json
import math

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, checkpoint_path


class Sieve9BAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("sieve-9b", model, revision, source, device)
        self.native = importlib.import_module("sieve")
        provenance = json.loads((self.path / "provenance.json").read_text())
        base_path, resolved = checkpoint_path(provenance["base_model"], provenance["base_revision"])
        self.engine = self.native.load_sieve(
            str(self.path), base=str(base_path), device=device, dtype=self.torch.bfloat16,
        )
        self.attention_model = self.engine.lm
        self.set_attention("sdpa")
        capacity = self.engine.lm.config.max_position_embeddings
        self.limit = 98304 if context_limit is None else context_limit
        if not 1 <= self.limit <= capacity:
            raise ValueError("Sieve-9B context limit exceeds checkpoint capacity")
        temperature = float(self.engine.head.temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid Sieve-9B released temperature")
        self.settings.update({
            "dtype": "bfloat16", "head_dtype": "float32", "temperature": temperature,
            "base_model": provenance["base_model"], "base_revision": resolved,
            "max_input_tokens": self.limit, "max_state_tokens": 65536,
            "max_question_tokens": 32768, "checkpoint_position_limit": capacity,
            "input_length_policy": "reject-overflow-native", "max_candidates": 255,
            "readout": "native-decision-index-single-causal-row-pointer-head",
            "renderer": "native-choice-authored-noul-numeric-score",
            "cuda_graphs": False, "attention_implementation": "sdpa",
        })

    def predict(self, case):
        wire = sifr_questions(case)
        predictions = []
        for question in case.questions:
            text, questions, _ = self.native.to_record(case.state, {"decision": wire[question.id]})
            record = self.engine.encode(text, questions, max_state=65536, max_branch=32768)
            if len(record["ids"]) > self.limit:
                raise ValueError("Complete Sieve-9B row exceeds context limit")
            state, _, rows = self.native.rows_of(record)
            row = rows[0]
            offset = record["n_state"]
            native_row = {"ids": state + row["ids"],
                          "opts": [offset + index for index in row["opts"]],
                          "decide": offset + row["decide"]}
            with self.torch.inference_mode():
                logits, _ = self.engine.row_logits([native_row])
                values = self.torch.softmax(logits[0].float() / self.engine.head.temperature, -1).tolist()
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions
