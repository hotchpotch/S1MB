"""Bespoke Nimble's pinned prompt, candidate codebook and calibration."""

import importlib
import json

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, state_text


def nimble_field(question):
    """Use anonymous field/choice names while retaining numeric score labels."""
    instruction = question.instructions
    if question.instructions_json is not None:
        instruction = json.dumps(json.loads(question.instructions_json), ensure_ascii=False)
    instruction = "\n\n".join(x for x in (question.system_prompt, instruction) if x)
    keys = [
        o.id
        if question.task == "noul"
        else f"level_{i}: {o.value}"
        if question.task == "score"
        else f"option_{i}"
        for i, o in enumerate(question.options)
    ]
    descriptions = {
        key: json.dumps(json.loads(o.description_json), ensure_ascii=False)
        if o.description_json is not None
        else o.description
        for key, o in zip(keys, question.options, strict=True)
    }
    # Explicit enums preserve authored boolean descriptions and arbitrary numeric
    # score levels; only the native integer expected-value convenience is unused.
    return keys, {
        "decision": {
            "type": "enum",
            "description": instruction,
            "choices": keys,
            "choice_descriptions": descriptions,
        }
    }


class NimbleAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("nimble", model, revision, source, device)
        self.native = importlib.import_module("inference")
        self.engine = self.native.NimbleModel(str(self.path))
        self.attention_model = self.engine.model
        original_limit = self.engine.contract["max_length"]
        if context_limit is not None:
            config = self.engine.model.config
            maximum = getattr(config, "text_config", config).max_position_embeddings
            if not 1 <= context_limit <= maximum:
                raise ValueError(f"Nimble context limit must be within 1..{maximum}")
            self.engine.contract = {**self.engine.contract, "max_length": context_limit}
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "max_input_tokens": self.engine.contract["max_length"],
            "checkpoint_runtime_limit": original_limit,
            "base_model": self.engine.contract["model"],
            "base_revision": self.engine.contract["revision"],
            "temperature": self.engine.temperature,
            "renderer": "native-enum-authored-levels-v1",
        }
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for q in case.questions:
            keys, schema = nimble_field(q)
            result = self.engine.score(state_text(case.state), schema)
            raw = result["fields"]["decision"]["probabilities"]
            if set(raw) != set(keys):
                raise ValueError("Nimble returned unexpected candidates")
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: raw[k] for o, k in zip(q.options, keys, strict=True)},
                )
            )
        return predictions
