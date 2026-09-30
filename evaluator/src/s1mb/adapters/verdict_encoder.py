"""Verdict's GLiClass runtime with explicit abstention conditioning."""

import importlib
import json
import math

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, state_text


class VerdictEncoderAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("verdict-encoder", model, revision, source, device)
        self.schema = importlib.import_module("core.primitives")
        self.formatting = importlib.import_module("core.formatting")
        native = importlib.import_module("core.engine_encoder")
        calibration = json.loads((self.path / "calibrator.json").read_text())
        self.engine = native.DecisionEngine(
            model_name_or_path=str(self.path), device=device, max_length=8192
        )
        if self.engine.calibrator is None:
            raise ValueError("Verdict failed to load its released calibration")
        self.attention_model = self.engine.model.model.encoder_model
        self.settings = {
            "dtype": "float32",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": 8192,
            "max_candidates": 24,
            "case_batch_size": 1,
            "calibration": calibration,
            "renderer": "native-choice-enum-authored-noul-numeric-score-v1",
            "probability_condition": "conditional-on-non-abstention",
            "extended_input_condition": "8192 instead of runtime 512 tokens",
        }
        self.set_attention("sdpa")

    def predict(self, case):
        predictions = []
        state = state_text(case.state)
        for q in case.questions:
            if len(q.options) > 24:
                raise ValueError("Verdict supports at most 24 candidates; refusing truncation")
            options = []
            for i, option in enumerate(q.options):
                description = option.description_json or option.description
                if q.task == "noul":
                    description = f"{option.id}: {description}"
                elif q.task == "score":
                    description = f"{option.value}: {description}"
                options.append(self.schema.Option(id=f"option_{i}", description=description))
            query = self.schema.Choice(
                id="decision",
                question="\n\n".join(
                    x for x in (q.system_prompt, q.instructions_json or q.instructions) if x
                ),
                options=tuple(options),
            )
            _, labels, _ = self.formatting.format_query(state, query)
            prompt = self.formatting.build_model_input(query.question, state, labels)
            if len(self.engine.tokenizer(prompt, truncation=False)["input_ids"]) > 8192:
                raise ValueError("Verdict input exceeds context limit; refusing truncation")
            raw = self.engine.evaluate(state, [query]).results[0].probabilities
            values = [raw[f"option_{i}"] for i in range(len(q.options))]
            mass = sum(values)
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError("Verdict returned no finite non-abstention probability mass")
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: p / mass for o, p in zip(q.options, values, strict=True)},
                )
            )
        return predictions
