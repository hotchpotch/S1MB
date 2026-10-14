"""Saihai 310M's native encoder heads with explicit native input budgeting."""

import importlib

from s1mb.data import Prediction, check_probabilities

from .bekko_v0 import native_input
from .upstream import UpstreamAdapter


class Saihai310mAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, *, native_adaptive_budget=False):
        self.native_adaptive_budget = native_adaptive_budget
        self.setup("saihai-310m", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        self.engine = transformers.AutoModel.from_pretrained(
            str(self.path), trust_remote_code=True, device=device, attn_implementation="sdpa",
        )
        self.runtime = self.engine.runtime
        self.native = importlib.import_module(type(self.runtime).__module__)
        self.settings.update({
            "loader": "AutoModel/BekkoModel", "dtype": "bfloat16-autocast",
            "attention_implementation": "sdpa",
            "input_length_policy": ("native-adaptive-v1-truncate"
                                    if native_adaptive_budget else "reject-overflow"),
            "query_truncation": self.runtime.query_truncation,
            "candidate_truncation": "right",
            "budget_allocation": "half-shares-lend-unused-candidate-odd-token",
            "context_length": self.runtime.context_length,
            "query_length": self.runtime.query_length,
            "document_length": self.runtime.document_length,
            "prefix_layout": self.runtime.prefix_layout,
            "renderer": "native-bekko-typed-input-anonymous-slots-v1",
            "microbatch_tokens": 64000,
        })

    def predict(self, case):
        request = native_input(case)
        mappings = []
        for index, (decision, question) in enumerate(zip(request["decisions"], case.questions, strict=True)):
            decision["id"] = f"question_{index}"
            mapping = {}
            for slot, criterion in enumerate(decision["criteria"]):
                original = criterion["id"]
                criterion["id"] = original if question.task == "noul" else f"option_{slot}"
                mapping[criterion["id"]] = original
            mappings.append(mapping)
        if not getattr(self, "native_adaptive_budget", False):
            self.check_input(request)
        result = self.engine.predict(request, show_progress_bar=False, token_budget=64000)
        if set(result) != {d["id"] for d in request["decisions"]}:
            raise ValueError("Native output has missing or unexpected decision IDs")
        predictions = []
        for index, (question, mapping) in enumerate(zip(case.questions, mappings, strict=True)):
            values = result[f"question_{index}"]["probabilities"]
            check_probabilities(values, list(mapping))
            probabilities = {original: values[slot] for slot, original in mapping.items()}
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=probabilities,
            ))
        return predictions

    def check_input(self, request):
        groups = self.native.input_groups(request, prefix_layout=self.runtime.prefix_layout)
        prepared = self.runtime.prepare_groups(groups)
        for group, actual in zip(groups, prepared, strict=True):
            # A character-derived upper bound retains every token in the native renderer.
            bound = 4 * (len(group.query) + sum(map(len, group.candidates))) + 1024
            query = self.runtime._queries([group.query], [group.query_parts], [bound])[0]
            documents = self.runtime._documents(
                group.candidates, [group.task] * len(group.candidates), bound,
            )
            if query != actual.query or documents != actual.documents:
                raise ValueError("Saihai 310M input exceeds native limits; refusing truncation")

    def close(self):
        del self.runtime
        super().close()
