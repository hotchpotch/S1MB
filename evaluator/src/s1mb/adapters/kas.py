"""Pinned Kas repeated-prompt scoring with complete authored criteria."""

import importlib

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


class KasAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Kas context_limit must be positive")
        self.setup("kas", model, revision, source, device)
        native = importlib.import_module("kas_engine.engine")
        self.engine = native.KasEngine(
            model=str(self.path), device=device, dtype="bfloat16", attn="sdpa",
            temperature=1.0, repeat_prompt=True, max_tokens=context_limit or 32768,
        )
        self.attention_model = self.engine.mdl
        if context_limit is not None and context_limit > self.engine.mdl.config.max_position_embeddings:
            raise ValueError("Kas context limit exceeds checkpoint capacity")
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa",
            "max_input_tokens": self.engine.limit, "input_length_policy": "reject-overflow-native",
            "temperature": 1.0, "repeat_prompt": True,
            "max_candidates": len(self.engine._labels), "questions_per_call": 1,
            "runtime": "native-repeated-json-label-logits-prefix-cache",
            "renderer": "native-choice-authored-noul-numeric-score",
        })

    def predict(self, case):
        predictions = []
        questions = sifr_questions(case)
        for question in case.questions:
            item = questions[question.id]
            if not 2 <= len(item["criteria"]) <= len(self.engine._labels):
                raise ValueError("Kas candidate count exceeds native label capacity")
            response, _ = self.engine(case.state, {"decision": item})
            answers = response["answers"]
            if set(answers) != {"decision"}:
                raise ValueError("Kas returned incorrect answer count")
            probabilities = answers["decision"]["probabilities"]
            keys = ( [option.id for option in question.options] if question.task == "noul"
                    else [f"option_{i}" for i in range(len(question.options))] )
            check_probabilities(probabilities, keys)
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id,
                probabilities={option.id: probabilities[key] for option, key in
                               zip(question.options, keys, strict=True)},
            ))
        return predictions

    def close(self):
        if hasattr(self, "engine"):
            self.engine._kv = None
            self.engine._kv_ids = []
        super().close()
