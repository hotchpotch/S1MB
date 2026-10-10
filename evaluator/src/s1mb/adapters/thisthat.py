"""Pinned this-that native answer-slot scoring without state truncation."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, state_text


def native_questions(case, question_type):
    rendered = questions_for_api(case.questions, structured=True)
    result = []
    for question in case.questions:
        item = rendered[question.id]
        instructions = item["instructions"]
        text = instructions if isinstance(instructions, str) else json.dumps(
            instructions, ensure_ascii=False
        )
        descriptions = []
        for index, option in enumerate(question.options):
            description = (
                item["criteria"][index] if question.task == "score"
                else item["criteria"][option.id]
            )
            if question.task == "score":
                description = {"value": option.value, "description": description}
            descriptions.append(
                description if isinstance(description, str)
                else json.dumps(description, ensure_ascii=False)
            )
        if len(set(descriptions)) != len(descriptions):
            # Native Question identifies options by their text. Anonymous
            # positions retain duplicate authored meanings as separate candidates.
            descriptions = [f"option_{i}: {value}" for i, value in enumerate(descriptions)]
        result.append(question_type(text, descriptions))
    return result


class ThisThatAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("this-that context_limit must be positive")
        self.setup("thisthat", model, revision, source, device)
        self.native = importlib.import_module("thisthat")
        backend_type = importlib.import_module("thisthat.backends.torch_backend").TorchBackend
        backend = backend_type.load(
            str(self.path), device=device, dtype="bfloat16", attn_implementation="sdpa",
        )
        self.engine = self.native.TypedDecider(backend)
        self.tokenizer = self.engine.tokenizer
        self.attention_model = self.engine.model
        self.context_limit = context_limit or 32768
        if self.context_limit > self.attention_model.config.get_text_config().max_position_embeddings:
            raise ValueError("this-that context limit exceeds checkpoint capacity")
        self.prompt = importlib.import_module("thisthat.prompt")
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa",
            "temperature": 1.0, "max_input_tokens": self.context_limit,
            "max_candidates": 255, "layout": "state_first",
            "input_length_policy": "full-state-reject-overflow-before-forward",
            "readout": "native-answer-slot-restricted-label-softmax",
            "renderer": "native-text-with-structured-instructions-and-numeric-score",
            "duplicate_description_policy": "anonymous-position-prefix-for-duplicate-questions",
        })

    def predict(self, case):
        state = state_text(case.state)
        native = native_questions(case, self.native.Question)
        predictions = []
        for question, request in zip(case.questions, native, strict=True):
            # Native state_first slices the combined prefix/state tokenization.
            # A budget at least that exact length preserves the complete state.
            state_budget = len(self.tokenizer.encode("Context:\n" + state, add_special_tokens=False))
            built = self.prompt.build(
                self.tokenizer, state, [request], max_state_tokens=state_budget,
            )
            padded = ((len(built["ids"]) + 63) // 64) * 64
            if padded > self.context_limit:
                raise ValueError(
                    f"this-that input requires {padded} padded tokens, limit "
                    f"{self.context_limit}; refusing truncation"
                )
            answer = self.engine.decide(state, request, max_state_tokens=state_budget)
            if tuple(answer.options) != tuple(request.options):
                raise ValueError("this-that returned different options")
            probabilities = {
                option.id: float(value)
                for option, value in zip(question.options, answer.probabilities, strict=True)
            }
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=probabilities,
            ))
        return predictions

    def close(self):
        if hasattr(self, "tokenizer") and hasattr(self, "prompt"):
            self.prompt._LABELS.pop(id(self.tokenizer), None)
        super().close()
