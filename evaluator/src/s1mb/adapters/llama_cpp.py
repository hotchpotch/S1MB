"""Native System One HTTP inference through a caller-managed llama-server."""

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx

from s1mb.data import InferenceCase, ModelInfo, Prediction

from .base import decode_answers, questions_for_api


class LlamaCppAdapter:
    """Connect to /v1/systemone without loading weights or managing a server."""

    def __init__(
        self,
        model: str,
        revision: str,
        *,
        base_url: str = "http://127.0.0.1:8080",
        served_model: str | None = None,
        timeout: float = 600,
        case_batch_size: int = 1,
        max_questions: int = 64,
        max_candidates: int | None = None,
        max_request_bytes: int = 16 * 1024 * 1024,
        retries: int = 2,
        runtime_settings: dict | None = None,
    ):
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP URL without credentials, query or fragment")
        if not model or not revision or served_model == "":
            raise ValueError("Specify the model identity and checkpoint revision")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if isinstance(retries, bool) or not isinstance(retries, int):
            raise TypeError("retries must be an integer")
        for name, value, maximum in [
            ("case_batch_size", case_batch_size, 32), ("max_questions", max_questions, 64),
            ("retries", retries + 1, 6), ("max_request_bytes", max_request_bytes, 64 * 1024 * 1024),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"Invalid {name}")
        if max_candidates is not None and (
            isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 2
        ):
            raise ValueError("max_candidates must be at least two")
        self.model, self.revision = model, revision
        self.base_url = str(url).rstrip("/") + "/"
        self.served_model = served_model
        self.resolved: str | None = None
        self.case_batch_size = case_batch_size
        self.max_questions = max_questions
        self.max_candidates = max_candidates
        self.max_request_bytes = max_request_bytes
        self.timeout, self.retries = timeout, retries
        self.runtime_settings = json.loads(json.dumps({} if runtime_settings is None else runtime_settings, allow_nan=False))
        if not isinstance(self.runtime_settings, dict):
            raise TypeError("runtime_settings must be an object")
        self._lock = Lock()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
        key = os.environ.get("S1MB_LLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    def predict(self, case: InferenceCase) -> list[Prediction]:
        if not 1 <= len(case.questions) <= self.max_questions:
            raise ValueError("Question count exceeds the configured server capacity")
        if len({q.id for q in case.questions}) != len(case.questions):
            raise ValueError("Duplicate question IDs")
        for q in case.questions:
            if self.max_candidates is not None and len(q.options) > self.max_candidates:
                raise ValueError("Candidate count exceeds the configured server capacity")
            if q.task == "score" and len(q.options) > 10:
                raise ValueError("Native llama.cpp Score supports at most ten levels")
        # Identifiers are transport keys, never source labels in model text.
        anonymous = case.model_copy(update={"questions": [
            q.model_copy(update={"id": f"question_{i}"}) for i, q in enumerate(case.questions)
        ]})
        body = {
            "state": case.state,
            "questions": questions_for_api(anonymous.questions, sort_score=True, structured=True,
                                            anonymous_choice=True),
        }
        if self.served_model:
            body["model"] = self.served_model
        content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        if len(content) > self.max_request_bytes:
            raise ValueError("Request exceeds max_request_bytes; refusing truncation")
        for attempt in range(self.retries + 1):
            with self._lock:
                self.usage["requests"] += 1
                self.usage["retries"] += int(attempt > 0)
            try:
                response = self.client.post("v1/systemone", content=content,
                                            headers={"Content-Type": "application/json"})
            except httpx.TransportError:
                if attempt == self.retries:
                    # Transport exceptions can contain URLs or credentials.
                    raise RuntimeError("llama.cpp transport failure") from None
                time.sleep(2**attempt)
                continue
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                break
            time.sleep(2**attempt)
        if response.is_error:
            raise RuntimeError(f"llama.cpp HTTP {response.status_code}")
        payload = response.json()
        resolved = payload["model"]
        if not isinstance(resolved, str) or not resolved:
            raise ValueError("Missing served model identity")
        if self.served_model is not None and resolved != self.served_model:
            raise ValueError("Served model differs from the configured alias")
        if payload.get("truncated"):
            raise ValueError("Server truncated the input")
        tokens = payload["usage"]
        if any(isinstance(tokens[k], bool) or not isinstance(tokens[k], int) or tokens[k] < 0
               for k in ("input_tokens", "output_tokens")):
            raise ValueError("Invalid server token usage")
        predictions = decode_answers(anonymous, payload["answers"], sort_score=True,
                                     anonymous_choice=True)
        with self._lock:
            if self.resolved is not None and self.resolved != resolved:
                raise ValueError("Served model changed during the run")
            self.resolved = resolved
            for k in ("input_tokens", "output_tokens"):
                self.usage[k] += tokens[k]
        return [prediction.model_copy(update={"question_id": original.id})
                for prediction, original in zip(predictions, case.questions, strict=True)]

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction] | Exception]:
        def predict_one(case: InferenceCase) -> list[Prediction] | Exception:
            try:
                return self.predict(case)
            except Exception as exc:  # noqa: BLE001 - preserve individual request failures
                return exc

        with ThreadPoolExecutor(max_workers=self.case_batch_size) as pool:
            return list(pool.map(predict_one, cases))

    def metadata(self) -> ModelInfo:
        return ModelInfo(id=self.model, adapter="llama-cpp", revision=self.revision, settings={
            "backend": "caller-managed-llama-server", "base_url": self.base_url,
            "endpoint": "/v1/systemone", "served_model_name": self.resolved or self.served_model,
            "revision_source": "caller-declared-checkpoint-revision",
            "runtime": dict(self.runtime_settings), "case_batch_size": self.case_batch_size,
            "max_questions": self.max_questions, "max_candidates": self.max_candidates,
            "max_request_bytes": self.max_request_bytes, "timeout_seconds": self.timeout,
            "max_retries": self.retries, "renderer": "native-systemone-structured-anonymous-v1",
            "score_criteria_order": "ascending-value-v1", "temperature": "model-file-native",
            "input_length_policy": "server-native-overflow-rejection-required-no-local-truncation",
            "probability_normalization": "native-server-no-client-renormalization",
        })

    def close(self) -> None:
        """Close this HTTP client; the external server is never terminated."""
        self.client.close()
