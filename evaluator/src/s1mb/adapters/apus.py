"""APUS high-effort choice readout with an explicit enum task bridge."""

import importlib
import string
from typing import Any, cast

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, state_text


class ApusAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("apus", model, revision, source, device)
        native = importlib.import_module("openjet_runtime")
        self.engine = native.OpenJet.from_pretrained(str(self.path), device, "bfloat16")
        self.attention_model = self.engine.model
        original_limit = self.engine.max_length
        if context_limit is not None:
            config = self.engine.model.config
            maximum = getattr(config, "text_config", config).max_position_embeddings
            if not 1 <= context_limit <= maximum:
                raise ValueError(f"APUS context limit must be within 1..{maximum}")
            self.engine.max_length = context_limit
        capacity = 16 if max_candidates is None else max_candidates
        if not 16 <= capacity <= 255:
            raise ValueError("APUS candidate limit must be within 16..255")
        contracts = cast(Any, importlib.import_module("openjet_runtime.contracts"))
        letters = list(string.ascii_uppercase)
        letters += [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        codes, seen = [], set()
        for letter in letters:
            ids = self.engine.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in seen:
                codes.append(letter)
                seen.add(ids[0])
        if codes[:16] != list("ABCDEFGHIJKLMNOP") or len(codes) < capacity:
            raise ValueError("APUS tokenizer cannot support the requested native-prefix codebook")
        contracts.LABELS = tuple(codes[:capacity])
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": self.engine.max_length,
            "max_candidates": capacity,
            "checkpoint_max_candidates": 16,
            "candidate_codes": codes[:capacity],
            "checkpoint_runtime_limit": original_limit,
            "case_batch_size": 1,
            "effort": "high",
            "renderer": "native-choice-enum-authored-noul-numeric-score-v1",
            "task_bridge": "all tasks use choice; native score_level is a binary proposition",
        }
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for q in case.questions:
            criteria = []
            for i, option in enumerate(q.options):
                description = option.description_json or option.description
                if q.task == "noul":
                    description = f"{option.id}: {description}"
                elif q.task == "score":
                    description = f"{option.value}: {description}"
                criteria.append({"id": f"option_{i}", "description": description})
            request = {
                "id": "decision",
                "group_id": "state",
                "primitive": "choice",
                "state": state_text(case.state),
                "instructions": "\n\n".join(
                    x for x in (q.system_prompt, q.instructions_json or q.instructions) if x
                ),
                "criteria": criteria,
            }
            raw = self.engine.decide(request, effort="high")["probabilities"]
            if set(raw) != {c["id"] for c in criteria}:
                raise ValueError("APUS returned unexpected candidates")
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: raw[f"option_{i}"] for i, o in enumerate(q.options)},
                )
            )
        return predictions
