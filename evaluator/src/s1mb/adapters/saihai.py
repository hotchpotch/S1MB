"""Saihai's released BF16 backend with its native input conversion."""

import importlib
import json
import sys

from s1mb.data import Prediction, check_probabilities

from .upstream import UpstreamAdapter


def saihai_input(case, question):
    """Preserve authored ordering, structured values and native Noul order."""
    options = list(question.options)
    if question.task == "noul":
        options.sort(key=lambda option: option.id == "true")
    item = {
        "state": case.state,
        "instructions": json.loads(question.instructions_json)
        if question.instructions_json else question.instructions,
        "type": question.task,
        "options": [
            {
                "label": option.id if question.task == "noul" else f"option_{index}",
                "description": json.loads(option.description_json)
                if option.description_json else option.description,
                **({"value": option.value} if question.task == "score" else {}),
            }
            for index, option in enumerate(options)
        ],
    }
    if question.system_prompt:
        item["instructions"] = {
            "system": question.system_prompt, "instruction": item["instructions"],
        }
    return item, options


class SaihaiAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        if sys.version_info < (3, 12):
            raise RuntimeError("The released Saihai runtime requires Python 3.12 or newer")
        self.setup("saihai", model, revision, source, device)
        native = importlib.import_module("saihai.model")
        self.engine = native.Saihai.from_pretrained(
            str(self.path), backend="transformers", device=device, dtype=self.torch.bfloat16,
        ).backend
        self.attention_model = self.engine.model
        contract = self.engine.contract
        if contract.get("prompt_version") != "saihai-pplx-json-v1":
            raise ValueError("Saihai requires its released native decision protocol")
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa",
            "max_input_tokens": contract["max_input_tokens"],
            "input_length_policy": "reject-overflow-native",
            "native_backend": "saihai.model.TorchDecisionModel",
            "renderer": "native-saihai-exact-input-v1",
            "prompt_version": contract["prompt_version"],
            "codebook": "pplx-decider", "candidate_limit": len(contract["codes"]),
            "score_level_limit": 10, "temperature": 1.0,
            "candidate_order": "authored; noul false,true",
            "base_model": contract["base"]["model_id"],
            "base_revision": contract["base"]["revision"],
            "generates_text": False,
        })

    def predict(self, case):
        predictions = []
        for question in case.questions:
            item, options = saihai_input(case, question)
            values = self.engine.predict(item)["probabilities"]
            probabilities = dict(zip((option.id for option in options), values, strict=True))
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=probabilities,
            ))
        return predictions
