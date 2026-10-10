"""Pinned llm2jev HF reference with explicit calibration and complete inputs."""

import importlib
import json
import math

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


def llm2jev_questions(case):
    questions = questions_for_api(case.questions, structured=True, anonymous_choice=True)
    for question in case.questions:
        if question.task == "score":
            item = questions[question.id]
            item["criteria"] = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, item["criteria"], strict=True)
            ]
    return questions


def llm2jev_state(state):
    """Preserve fields on records that the native renderer would mistake for chat."""
    messages = state["messages"] if isinstance(state, dict) and set(state) == {"messages"} else state
    if (
        isinstance(messages, list) and messages
        and all(isinstance(message, dict) and "role" in message for message in messages)
        and any(
            set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or message["role"] not in {"system", "user", "assistant", "tool"}
            for message in messages
        )
    ):
        return json.dumps(state, ensure_ascii=False)
    return state


class BoundedHF:
    """Check complete native tokenization before the unmodified HF scorer runs."""

    def __init__(self, backend, limit):
        self.backend, self.limit = backend, limit

    def score(self, text, images, ids):
        if images:
            raise ValueError("These benchmark runs accept text-only llm2jev requests")
        if len(self.backend.tok.encode(text, add_special_tokens=False)) > self.limit:
            raise ValueError("llm2jev input exceeds context limit; refusing truncation")
        return self.backend.score(text, images, ids)


class LLM2JevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, temperature, context_limit=None):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("llm2jev temperature must be finite and positive")
        if context_limit is not None and context_limit < 1:
            raise ValueError("llm2jev context_limit must be positive")
        self.setup("llm2jev", model, revision, source, device)
        native = importlib.import_module("llm2jev.engine")
        backend_module = importlib.import_module("llm2jev.backends")
        backend = backend_module.HF(str(self.path), device=device, dtype="bfloat16")
        self.attention_model = backend.model
        capacity = self.attention_model.config.get_text_config().max_position_embeddings
        limit = context_limit or 32768
        if limit > capacity:
            raise ValueError("llm2jev context limit exceeds checkpoint capacity")
        self.set_attention("sdpa")
        self.engine = native.LLM2Jev(
            backend.processor or backend.tok, BoundedHF(backend, limit),
            temperature=temperature, style="chat", workers=1,
        )
        self.settings.update({
            "dtype": "bfloat16",
            "max_input_tokens": limit,
            "input_length_policy": "reject-overflow",
            "questions_per_call": 1,
            "temperature": temperature,
            "prompt_style": "native-chat-thinking-disabled",
            "candidate_slots": len(self.engine.labels),
            "runtime": "native-hf-reference-full-vocab-last-position",
            "structured_state_policy": "native-chat-or-json-for-role-records-with-extra-fields",
            "renderer": "native-structured-anonymous-numeric-score-v1",
        })

    def predict(self, case):
        answers = {}
        for key, question in llm2jev_questions(case).items():
            response = self.engine(llm2jev_state(case.state), {"decision": question})
            if set(response) != {"decision"}:
                raise ValueError("llm2jev returned incorrect answer count")
            answers[key] = response["decision"]
        return decode_answers(case, answers, anonymous_choice=True)

    def close(self):
        if hasattr(self, "engine"):
            self.engine.pool.shutdown(wait=True, cancel_futures=True)
        super().close()
