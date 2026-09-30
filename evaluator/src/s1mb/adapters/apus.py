"""APUS high-effort choice readout with an explicit enum task bridge."""

import importlib

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, state_text


class ApusAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("apus", model, revision, source, device)
        native = importlib.import_module("openjet_runtime")
        self.engine = native.OpenJet.from_pretrained(str(self.path), device, "bfloat16")
        self.attention_model = self.engine.model
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": self.engine.max_length,
            "max_candidates": 16,
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
