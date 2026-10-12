"""Pinned Sieve native prefix forks and calibrated scalar readout, without trimming."""

import importlib
import json
import math

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, checkpoint_path, state_text


def sieve_questions(case):
    rendered = questions_for_api(case.questions, structured=True)
    result = []
    for question in case.questions:
        item = rendered[question.id]
        descriptions = []
        for i, option in enumerate(question.options):
            description = (item["criteria"][i] if question.task == "score"
                           else item["criteria"][option.id])
            if question.task == "score":
                description = {"value": option.value, "description": description}
            descriptions.append(state_text(description))
        if any(not description for description in descriptions):
            raise ValueError("Sieve requires nonempty native option descriptions")
        result.append((state_text(item["instructions"]), descriptions))
    return result


class SieveAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, base_revision, context_limit=None):
        if not base_revision:
            raise ValueError("Sieve requires an explicit base revision")
        if context_limit is not None and context_limit < 1:
            raise ValueError("Sieve context_limit must be positive")
        self.setup("sieve", model, revision, source, device)
        self.native = importlib.import_module("modeling_sieve")
        config = json.loads((self.path / "config.json").read_text())
        base_path, resolved = checkpoint_path(config["base_model"], base_revision)
        loader = importlib.import_module("safetensors.torch").load_file
        self.engine = self.native.Sieve(
            str(base_path), config["read_layer"], loader(self.path / "adapter_model.safetensors"),
            loader(self.path / "head.safetensors"), config["list_options"],
            T=config.get("temperature", 1.0), lora_config=config["lora"],
            device=device, dtype=self.torch.float32, max_len=config.get("max_len", 2048),
        )
        if not math.isfinite(self.engine.T) or self.engine.T <= 0:
            raise ValueError("Invalid native Sieve temperature")
        if context_limit is not None:
            if context_limit > self.engine.max_len:
                raise ValueError("Sieve context_limit exceeds native release capacity")
            self.engine.max_len = context_limit
        self.attention_model = self.engine.model
        self.set_attention("sdpa")
        self.settings.update({
            "dtype": "float32", "base_model": config["base_model"], "base_revision": resolved,
            "read_layer": config["read_layer"], "temperature": self.engine.T,
            "max_input_tokens": self.engine.max_len, "list_options": self.engine.list_options,
            "input_length_policy": "reject-overflow-before-native-prefix-no-trimming",
            "branch_batch_size": 8, "branch_padded_token_budget": 32768,
            "readout": "native-isolated-prefix-forks-scalar-head-global-softmax",
            "typed_mapping": "authored-option-descriptions-including-numeric-score",
            "renderer": "native-structured-numeric-score-v1",
        })

    def _probabilities(self, state, instructions, options):
        question = self.native.listed_question(instructions, options) if self.engine.list_options else instructions
        prefix = self.engine._ids(self.native.choice_prefix(state, question))
        branches = [self.engine._ids(option) for option in options]
        if any(not branch for branch in branches):
            raise ValueError("Sieve option tokenization is empty")
        maximum = len(prefix) + max(map(len, branches))
        if maximum > self.engine.max_len:
            raise ValueError(f"Sieve input requires {maximum} tokens; refusing truncation")
        # Native fork mutates/reorders its cache. Recompute the same prefix for
        # each bounded group, retaining every candidate and one global softmax.
        size = min(8, max(1, 32768 // maximum))
        logits = []
        with self.torch.inference_mode():
            for start in range(0, len(branches), size):
                group = branches[start:start + size]
                cache, attention = self.engine._prefix_pass([prefix])
                hidden = self.engine._fork_pass(
                    cache, attention, [len(prefix)], group, [0] * len(group),
                )
                logits.append((self.engine.head(hidden) / self.engine.T).float().cpu())
                del cache, attention, hidden
            return self.torch.softmax(self.torch.cat(logits), dim=-1).tolist()

    def predict(self, case):
        predictions = []
        for question, (instructions, options) in zip(
            case.questions, sieve_questions(case), strict=True,
        ):
            probabilities = dict(zip(
                [option.id for option in question.options],
                self._probabilities(state_text(case.state), instructions, options), strict=True,
            ))
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=probabilities,
            ))
        return predictions
