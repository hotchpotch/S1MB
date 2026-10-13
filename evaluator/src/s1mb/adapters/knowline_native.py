"""Pinned KnowLine HF reference with complete native inputs and explicit CUDA."""

import importlib.util

from .base import decode_answers
from .llm2jev import BoundedHF, llm2jev_questions, llm2jev_state
from .upstream import UpstreamAdapter


class KnowLineNativeAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("knowline", model, revision, source, device)
        spec = importlib.util.spec_from_file_location(
            "s1mb_pinned_knowline_server", self.source / "knowline_server.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError("Source must contain the native knowline_server.py")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        backend = native.HF(str(self.path), device=device, dtype="bfloat16")
        self.attention_model = backend.model
        capacity = self.attention_model.config.get_text_config().max_position_embeddings
        limit = context_limit or min(32768, capacity)
        if not 1 <= limit <= capacity:
            raise ValueError("KnowLine context limit exceeds checkpoint capacity")
        self.set_attention("sdpa")
        if self.attention_model.config.get_text_config().model_type == "qwen3_5_text":
            self.enable_kernels()
        self.engine = native.KnowLine(
            backend.tok, BoundedHF(backend, limit), temperature=1.0, workers=1
        )
        self.settings.update({
            "dtype": "bfloat16", "max_input_tokens": limit,
            "input_length_policy": "reject-overflow", "temperature": 1.0,
            "candidate_slots": len(self.engine.labels),
            "prompt_style": "native-chat-thinking-disabled",
            "runtime": "knowline-native-hf-reference-full-vocab-last-position",
            "renderer": "native-structured-anonymous-numeric-score-v1",
            "structured_state_policy": "native-chat-or-json-for-role-records-with-extra-fields",
        })

    def predict(self, case):
        answers = {}
        for key, question in llm2jev_questions(case).items():
            response, _usage = self.engine.run(llm2jev_state(case.state), {"decision": question})
            if set(response) != {"decision"}:
                raise ValueError("KnowLine returned incorrect answer count")
            answers[key] = response["decision"]
        return decode_answers(case, answers, anonymous_choice=True)

    def close(self):
        if hasattr(self, "engine"):
            self.engine.pool.shutdown(wait=True, cancel_futures=True)
        super().close()
