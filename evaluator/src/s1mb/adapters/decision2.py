"""Pinned native Decision 2.0 package with exact, bounded typed-question inference."""

import importlib
import shutil
import tempfile
from pathlib import Path

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


class Decision2InputTooLong(ValueError):
    """The pinned native runtime rejected a complete input without truncation."""


def decision2_questions(case):
    questions = questions_for_api(case.questions, structured=True, anonymous_choice=True)
    for question in case.questions:
        if question.task == "score":
            item = questions[question.id]
            item["criteria"] = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, item["criteria"], strict=True)
            ]
    return questions


class Decision2Adapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Decision 2.0 context_limit must be positive")
        self.setup("decision2", model, revision, source, device)
        self.native = importlib.import_module("decision2")
        self.bundle_directory = None
        native_path = self.path
        # The native verifier forbids links, including Hub snapshot blob links.
        # Keep the resolved Hub revision while verifying an independent copy.
        if any(path.is_symlink() for path in self.path.rglob("*")):
            self.bundle_directory = tempfile.TemporaryDirectory(prefix="s1mb-decision2-")
            native_path = Path(self.bundle_directory.name) / "bundle"
            shutil.copytree(self.path, native_path, symlinks=False)
        self.engine = self.native.Decision2.from_pretrained(
            str(native_path),
            device=device,
            bf16_resident=True,
            graphs=False,
            kernels=False,
            share_context=False,
        )
        if context_limit is not None:
            if context_limit > self.engine.max_input_tokens:
                raise ValueError("Decision 2.0 context_limit exceeds native release capacity")
            self.engine.backend.cap = context_limit
        self.attention_model = self.engine.backend.model.backbone
        if self.attention_model.config._attn_implementation != "sdpa":
            self.set_attention("sdpa")
        self.settings.update(
            {
                "dtype": "native-bfloat16-exact-linear-residency-float32-head",
                "attention_implementation": "sdpa",
                "max_input_tokens": context_limit or self.engine.max_input_tokens,
                "release_max_input_tokens": self.engine.max_input_tokens,
                "input_length_policy": "reject-overflow-native",
                "questions_per_call": 1,
                "share_context": False,
                "graphs": False,
                "fused_kernels": False,
                "native_profile": self.engine.manifest["profile"],
                "native_temperatures": self.engine.backend.temperatures,
                "native_score_bias": self.engine.backend.score_bias,
                "native_manifest_identity": self.engine.manifest["identity"],
                "native_base": self.engine.manifest.get("base"),
                "renderer": "native-structured-anonymous-numeric-score-v1",
                "bundle_materialization": "independent-copy" if self.bundle_directory else "none",
            }
        )

    def predict(self, case):
        answers = {}
        for key, question in decision2_questions(case).items():
            response = self.engine.system_one(state=case.state, questions={"decision": question})
            native = response.get("answers")
            if not isinstance(native, dict) or set(native) != {"decision"}:
                raise ValueError("Decision 2.0 returned incorrect answer count")
            answer = native["decision"]
            if not isinstance(answer, dict):
                raise TypeError("Decision 2.0 returned malformed answer")
            reason = answer.get("error")
            if reason == "max_length_exceeded":
                raise Decision2InputTooLong(
                    "Native input exceeds release budget; refusing truncation"
                )
            if reason:
                raise ValueError(f"Decision 2.0 native answer error: {reason}")
            answers[key] = answer
        return decode_answers(case, answers, anonymous_choice=True)

    def close(self):
        super().close()
        if self.bundle_directory is not None:
            self.bundle_directory.cleanup()
