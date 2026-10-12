"""Two-stage target-free reasoning context for Meta Encoder decisions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from s1mb.data import InferenceCase, ModelInfo, Prediction, Question, check_probabilities

from .meta_encoder import MetaEncoderAdapter, _parsed_text, _state_text, render_query

REASONER_ID = "meta-models/Muse-Glimmer-30B"
REASONER_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
REASONER_PROMPT_PREFIX = (
    "Solve the decision below using careful step-by-step reasoning. Use only the supplied "
    "state, task, criteria, and option IDs. Do not mention the final option anywhere in the "
    "reasoning. After the reasoning, write exactly one separate line in the form "
    "FINAL_ANSWER: <option ID>."
)
FINAL_MARKER = re.compile(r"(?im)^\s*(?:FINAL_ANSWER|FINAL ANSWER)\s*:")
CONCLUSION = re.compile(
    r"(?im)^\s*(?:therefore|thus|hence|so,?)?\s*(?:the\s+)?"
    r"(?:correct\s+)?(?:answer|option|choice)\s+(?:is|would be|:)\s*"
)
VERIFIER_CANDIDATES = (
    "The proposed answer is correct under the stated task and criteria.",
    "The proposed answer is incorrect under the stated task and criteria.",
)
CONTEXT_FORMAT_VERSION = 1
MAX_CONTEXT_FILE_BYTES = 512 * 1024 * 1024
MAX_CONTEXT_RECORDS = 100_000
MAX_CONTEXT_TEXT_CHARS = 131_072


def load_reasoning_contexts(path: Path) -> tuple[dict[str, Any], str]:
    """Read a bounded artifact once; validate it and hash the exact consumed bytes."""
    if not path.is_file():
        raise ValueError("reasoning context artifact must be a regular file")
    with path.open("rb") as stream:
        file_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("reasoning context artifact must be a regular file")
        if file_stat.st_size > MAX_CONTEXT_FILE_BYTES:
            raise ValueError("reasoning context artifact exceeds the 512 MiB size limit")
        raw = stream.read(MAX_CONTEXT_FILE_BYTES + 1)
    if len(raw) > MAX_CONTEXT_FILE_BYTES:
        raise ValueError("reasoning context artifact exceeds the 512 MiB size limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ValueError("reasoning context artifact must contain valid UTF-8 JSON") from error
    validate_reasoning_payload(payload)
    return payload, hashlib.sha256(raw).hexdigest()


def validate_reasoning_payload(payload: Any) -> None:
    """Reject malformed records before any model is loaded."""
    if not isinstance(payload, dict):
        raise TypeError("reasoning context artifact must be a JSON object")
    version = payload.get("format_version")
    if type(version) is not int or version != CONTEXT_FORMAT_VERSION:
        raise ValueError("unsupported reasoning context format version")
    reasoner = payload.get("reasoner")
    if not isinstance(reasoner, dict):
        raise TypeError("reasoning context reasoner must be an object")
    if reasoner.get("id") != REASONER_ID or reasoner.get("revision") != REASONER_REVISION:
        raise ValueError("reasoning context checkpoint identity does not match the recipe")
    records = payload.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_CONTEXT_RECORDS:
        raise ValueError("reasoning context records must contain 1 to 100000 entries")
    for index, record in enumerate(records):
        prefix = f"reasoning context record {index}"
        if not isinstance(record, dict):
            raise TypeError(f"{prefix} must be an object")
        for field in ("case_id", "question_id"):
            value = record.get(field)
            if not isinstance(value, str) or not 1 <= len(value) <= 1024:
                raise ValueError(f"{prefix} has an invalid {field}")
        query_hash = record.get("query_sha256")
        if not isinstance(query_hash, str) or re.fullmatch(r"[0-9a-f]{64}", query_hash) is None:
            raise ValueError(f"{prefix} has an invalid query_sha256")
        if record.get("status") not in ("usable", "missing_boundary", "ambiguous_boundary"):
            raise ValueError(f"{prefix} has an invalid status")
        for field in ("raw_generation", "truncated_reasoning"):
            value = record.get(field)
            if not isinstance(value, str) or len(value) > MAX_CONTEXT_TEXT_CHARS:
                raise ValueError(f"{prefix} has an invalid or oversized {field}")
        if record["status"] == "usable" and not record["truncated_reasoning"].strip():
            raise ValueError(f"{prefix} has an empty usable reasoning prefix")
        reasoning_context(record)


def verification_queries(case: InferenceCase, question: Question, record: dict[str, Any]) -> list[str]:
    """Render every proposed answer with the decision's shared frozen rationale."""
    criteria = "\n".join(
        f"- {o.id}: {_parsed_text(o.description_json, o.description)}" for o in question.options
    )
    body = "\n".join(
        [
            "Determine whether the proposed answer is the correct answer to the stated task under its criteria.",
            f"Task: {_parsed_text(question.instructions_json, question.instructions)}",
            f"Input: {_state_text(case)}",
            "Criteria:",
            criteria,
        ]
    )
    prefix = f"Reasoning context:\n{reasoning_context(record)}\n\n{body}"
    return [
        f"{prefix}\nProposed answer: {o.id}: {_parsed_text(o.description_json, o.description)}"
        for o in question.options
    ]


def verification_probabilities(pairs: list[list[float]], temperature: float = 0.0) -> list[float]:
    """Select the largest FP32 verification gap, breaking ties by option order."""
    if not math.isfinite(temperature) or temperature != 0:
        raise ValueError("candidate_verify temperature is fixed at zero")
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("verification requires correct/incorrect cosine pairs")
    gaps = [struct.unpack("<f", struct.pack("<f", pair[0] - pair[1]))[0] for pair in pairs]
    if any(not math.isfinite(gap) for gap in gaps):
        raise ValueError("nonfinite verification gap")
    winner = max(range(len(gaps)), key=gaps.__getitem__)
    return [float(index == winner) for index in range(len(gaps))]


def reasoner_prompt(query: str) -> str:
    """Build the target-free generation prompt from the official rendered query."""
    return f"{REASONER_PROMPT_PREFIX}\n\n{query}"


def parse_reasoning_generation(raw: str) -> dict[str, Any]:
    """Cut at the earliest final-answer marker or answer-like conclusion."""
    markers = list(FINAL_MARKER.finditer(raw))
    conclusions = list(CONCLUSION.finditer(raw))
    boundaries = [(match.start(), "final_answer_marker") for match in markers]
    boundaries.extend((match.start(), "answer_like_conclusion") for match in conclusions)
    boundaries.sort()
    if not boundaries:
        return {
            "status": "missing_boundary",
            "truncated_reasoning": "",
            "cut_offset": None,
            "cut_rule": None,
            "answer_like_conclusion_before_marker": False,
            "boundary_counts": {"marker": 0, "conclusion": 0},
        }
    offset, rule = boundaries[0]
    prefix = raw[:offset].strip()
    first_marker = markers[0].start() if markers else None
    early_conclusion = bool(
        conclusions and (first_marker is None or conclusions[0].start() < first_marker)
    )
    retained_signal = FINAL_MARKER.search(prefix) or CONCLUSION.search(prefix)
    status = "usable" if prefix and retained_signal is None else "ambiguous_boundary"
    return {
        "status": status,
        "truncated_reasoning": prefix if status == "usable" else "",
        "cut_offset": offset,
        "cut_rule": rule,
        "answer_like_conclusion_before_marker": early_conclusion,
        "boundary_counts": {"marker": len(markers), "conclusion": len(conclusions)},
    }


def reasoning_context(record: dict[str, Any]) -> str:
    """Use a clean prefix when available, otherwise retain the nonempty generation."""
    prefix = str(record.get("truncated_reasoning", "")).strip()
    if record.get("status") == "usable" and prefix:
        return prefix
    raw = str(record.get("raw_generation", "")).strip()
    if not raw:
        raise ValueError("No generated reasoning context is available")
    return raw


def contextual_query(query: str, record: dict[str, Any]) -> str:
    return f"Reasoning context:\n{reasoning_context(record)}\n\n{query}"


def composite_parameter_metadata(encoder: ModelInfo, reasoner: dict[str, Any]) -> dict[str, Any]:
    """Sum independently loaded reasoner and encoder checkpoints when both are verified."""
    reasoner_parameters = reasoner.get("parameters")
    if (
        encoder.total_params is None
        or encoder.active_params is None
        or encoder.parameter_count_method != "embedding_excluded_parameters_v1"
        or not isinstance(reasoner_parameters, dict)
        or reasoner_parameters.get("parameter_count_method")
        != "embedding_excluded_parameters_v1"
        or not isinstance(reasoner_parameters.get("total_params"), int)
        or not isinstance(reasoner_parameters.get("active_params"), int)
    ):
        return {"total_params": None, "active_params": None, "parameter_count_method": None}
    return {
        "total_params": encoder.total_params + reasoner_parameters["total_params"],
        "active_params": encoder.active_params + reasoner_parameters["active_params"],
        "parameter_count_method": "embedding_excluded_parameters_v1",
    }


class MetaEncoderThinkAdapter(MetaEncoderAdapter):
    """Score official queries augmented with frozen, target-free reasoning contexts."""

    def __init__(
        self, *args, reasoning_contexts: Path, scoring: str = "cosine",
        temperature: float | None = None, **kwargs
    ) -> None:
        if scoring not in {"cosine", "candidate_verify"}:
            raise ValueError("unknown MetaEncoder-think scoring strategy")
        self.scoring = scoring
        if temperature is None:
            if scoring != "candidate_verify":
                raise ValueError("cosine scoring requires an explicit positive temperature")
            temperature = 0.0
        self._validate_temperature(temperature)
        payload, self._context_sha256 = load_reasoning_contexts(reasoning_contexts)
        self._context_format_version = payload["format_version"]
        reasoner = payload["reasoner"]
        records = payload["records"]
        self._contexts = {}
        for record in records:
            key = (record["case_id"], record["question_id"], record["query_sha256"])
            if key in self._contexts and reasoning_context(self._contexts[key]) != reasoning_context(record):
                raise ValueError("conflicting reasoning contexts for one rendered query")
            self._contexts[key] = record
        self._reasoner = {
            key: reasoner[key]
            for key in (
                "id",
                "revision",
                "dtype",
                "attention",
                "effective_device_map",
                "generation",
                "prompt_policy",
                "truncation_policy",
                "parameters",
            )
            if key in reasoner
        }
        self._context_counts = Counter(record.get("status") for record in records)
        super().__init__(*args, temperature=temperature, **kwargs)

    def _validate_temperature(self, temperature: float) -> None:
        if self.scoring == "candidate_verify":
            if not math.isfinite(temperature) or temperature != 0:
                raise ValueError("candidate_verify temperature is fixed at zero")
        else:
            super()._validate_temperature(temperature)

    def _render_query(self, case: InferenceCase, question: Question) -> str:
        query = render_query(case, question)
        key = (case.case_id, question.id, hashlib.sha256(query.encode()).hexdigest())
        if key not in self._contexts:
            raise ValueError("missing reasoning context for rendered query")
        return contextual_query(query, self._contexts[key])

    def predict(self, case: InferenceCase) -> list[Prediction]:
        if self.scoring == "cosine":
            return super().predict(case)
        verifier = self._candidate_embeddings(list(VERIFIER_CANDIDATES))
        predictions = []
        for question in case.questions:
            query = render_query(case, question)
            key = (case.case_id, question.id, hashlib.sha256(query.encode()).hexdigest())
            if key not in self._contexts:
                raise ValueError("missing reasoning context for rendered query")
            requests = verification_queries(case, question, self._contexts[key])
            # Match the measured length buckets and preserve authored option order.
            lengths = [
                len(self.processor(text=[self._chat(text)], padding=False, truncation=False,
                    add_special_tokens=False)["input_ids"][0])
                for text in requests
            ]
            if max(lengths) > self.context_limit:
                raise ValueError("Meta Encoder verification input exceeds context limit")
            vectors = {}
            for low, high, initial_batch in ((0, 4096, 8), (4096, 8192, 2), (8192, self.context_limit + 1, 1)):
                order = sorted((i for i, n in enumerate(lengths) if low <= n < high),
                               key=lambda i: (lengths[i], requests[i]))
                offset, batch_size = 0, initial_batch
                while offset < len(order):
                    batch = order[offset:offset + batch_size]
                    try:
                        embeddings = self._encode([requests[i] for i in batch], len(batch))
                    except RuntimeError as error:
                        retryable = isinstance(error, self.torch.cuda.OutOfMemoryError) or (
                            "mha_graph.execute" in str(error) and "is_good()" in str(error)
                        )
                        if not retryable or batch_size == 1:
                            raise
                        self.torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        continue
                    for i, embedding in zip(batch, embeddings, strict=True):
                        vectors[i] = embedding
                    offset += len(batch)
            pairs = (self.torch.stack([vectors[i] for i in range(len(requests))]) @ verifier.T).cpu().tolist()
            probabilities = dict(zip([o.id for o in question.options],
                verification_probabilities(pairs, self.temperature), strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions

    def metadata(self) -> ModelInfo:
        base = super().metadata()
        settings = dict(base.settings)
        settings["native_encoder_parameter_count"] = {
            "total_params": base.total_params,
            "active_params": base.active_params,
            "parameter_count_method": base.parameter_count_method,
        }
        settings.update(
            {
                "renderer": "text-choice-option-id-with-reasoning-context-v1",
                "reasoner": self._reasoner,
                "reasoning_prompt_policy": (
                    "target-free-official-query-reasoning-then-final-answer-v1"
                ),
                "reasoning_context_policy": (
                    "earliest-answer-boundary-else-entire-nonempty-generation-v1"
                ),
                "reasoning_context_status_counts": dict(self._context_counts),
                "reasoning_context_artifact": {
                    "sha256": self._context_sha256,
                    "format_version": self._context_format_version,
                },
                "fallback_contexts": sum(
                    count for status, count in self._context_counts.items() if status != "usable"
                ),
            }
        )
        settings["think_scoring"] = self.scoring
        if self.scoring == "candidate_verify":
            settings.update({
                "renderer": "shared-frozen-reasoning-per-candidate-verification-v1",
                "verification_candidates": list(VERIFIER_CANDIDATES),
                "verification_score": "float32-cosine-correct-minus-incorrect",
                "probability_rule": "one-hot-verification-argmax",
                "probability_computation": "one-hot-argmax-from-float32-verification-gaps",
                "verification_tie_break": "first-declared-option",
                "verification_batch_size": 8,
                "long_verification_batch_sizes": [2, 1],
            })
        return base.model_copy(
            update={
                "adapter": "meta-encoder-think",
                "settings": settings,
                **composite_parameter_metadata(base, self._reasoner),
            }
        )
