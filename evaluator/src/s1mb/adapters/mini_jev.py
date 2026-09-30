"""Mini-Jev's released NF4 backbone and joint candidate head."""

import importlib
import json
from types import FunctionType, MethodType

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, state_text


def full_summary_branch(native, tokenizer, state, question, question_type, option, max_tokens):
    """Keep the native token boundaries and mask while admitting the complete summary."""
    if set(state) != {"summary"}:
        raise ValueError("The extended Mini-Jev encoder requires summary-only state")
    ids, mask = native.encode_branch(
        tokenizer, {"summary": ""}, question, question_type, option, 8192
    )
    prefix = (
        native._encode(tokenizer, "<STATE>\n")
        + native._encode(tokenizer, native._section("SYSTEM", ""))
        + native._encode(tokenizer, native._section("USER_GOAL", ""))
        + native._encode(tokenizer, "<SUMMARY>\n")
    )
    if ids[: len(prefix)] != prefix:
        raise ValueError("Mini-Jev native summary boundary changed")
    summary = native._encode(tokenizer, native._text(state["summary"]))
    if len(ids) + len(summary) > max_tokens:
        raise ValueError("Mini-Jev input exceeds context limit; refusing truncation")
    at = len(prefix)
    return ids[:at] + summary + ids[at:], mask[:at] + [False] * len(summary) + mask[at:]


class MiniJevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("mini-jev", model, revision, source, device)
        self.native = importlib.import_module("inference")
        self.engine = self.native.MiniJev.load(str(self.path), device=device)
        self.attention_model = self.engine.backbone
        original_limit = self.engine.max_tokens
        if context_limit is not None:
            maximum = self.attention_model.config.max_position_embeddings
            if not 1 <= context_limit <= maximum:
                raise ValueError(f"Mini-Jev context limit must be within 1..{maximum}")
            self.engine.max_tokens = context_limit
            # Clone the native method's globals for this instance only. Its GPU
            # forward, suffix pooling and joint head remain the released code.
            method = self.engine.predict.__func__

            def encode(*args, **kwargs):
                return full_summary_branch(self.native, *args, **kwargs)

            extended = FunctionType(
                method.__code__,
                {**method.__globals__, "encode_branch": encode},
                method.__name__,
                method.__defaults__,
                method.__closure__,
            )
            extended.__kwdefaults__ = method.__kwdefaults__
            self.engine.predict = MethodType(extended, self.engine)
        config = json.loads((self.path / "config.json").read_text())
        self.settings = {
            "dtype": "nf4-double-quantization-bfloat16-compute-float32-head",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": self.engine.max_tokens,
            "checkpoint_runtime_limit": original_limit,
            "case_batch_size": 1,
            "base_model": config["base_model"],
            "base_revision": config["base_revision"],
            "renderer": "native-all-state-in-summary-anonymous-options-v1",
        }

    def predict(self, case):
        predictions = []
        # Arbitrary S1MB state fields must not disappear in the tool-state whitelist.
        state = {"summary": state_text(case.state)}
        summary_tokens = self.native._encode(
            self.engine.tokenizer, self.native._text(state["summary"])
        )
        for q in case.questions:
            instruction = "\n\n".join(
                x for x in (q.system_prompt, q.instructions_json or q.instructions) if x
            )
            options = [
                {
                    "id": f"option_{i}",
                    "type": q.task,
                    "label": o.id
                    if q.task == "noul"
                    else str(o.value)
                    if q.task == "score"
                    else f"option_{i}",
                    "description": o.description_json or o.description,
                }
                for i, o in enumerate(q.options)
            ]
            for option in options:
                empty, _ = self.native.encode_branch(
                    self.engine.tokenizer,
                    {"summary": ""},
                    instruction,
                    q.task,
                    option,
                    8192,
                )
                if len(empty) + len(summary_tokens) > self.engine.max_tokens:
                    raise ValueError("Mini-Jev input exceeds context limit; refusing truncation")
            result = self.engine.predict(state, instruction, q.task, options)
            raw = {o["id"]: o["probability"] for o in result["options"]}
            if set(raw) != {o["id"] for o in options}:
                raise ValueError("Mini-Jev returned unexpected candidates")
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: raw[f"option_{i}"] for i, o in enumerate(q.options)},
                )
            )
        return predictions
