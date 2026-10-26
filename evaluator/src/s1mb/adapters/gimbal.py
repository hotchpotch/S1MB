"""Gimbal-27B through its released, caller-managed `gimbal_serve` vLLM server.

The released helper (`helper/` in Superwheat/Gimbal-27B) serves the merged checkpoint at
POST /v1/systemone. Start it with `--full-precision`; the adapter never starts, stops or
configures that server.

The server scores every question independently: its premise is that question's instructions
followed by the state, and each candidate is a separate hypothesis branch. Its typed contract
differs from the TypeSafe contract in four places, each handled without loss and recorded in the
model settings:

- Noul: the native Noul type accepts a question without outcome definitions. To keep the authored
  true/false meanings, every Noul question is sent as an explicit two-option Choice whose options
  are the authored definitions in declared order (`explicit-choice-v1`).
- State: the server accepts a string or an object. Other JSON values are sent as their JSON text,
  the text the server renders for an object state.
- Values: instructions and descriptions may be strings, objects, arrays or null. Other JSON
  scalars are sent as their JSON text.
- Capacity: a request holds at most 64 questions. Larger cases are sent in consecutive chunks of
  64 with the same state.

Choice keys are anonymous and Score levels are sent in ascending numeric order, then mapped back
to the declared option IDs. Probabilities are the server's full-precision calibrated
`raw_probabilities`, never its two-decimal display rounding, and are not renormalized. The server
rejects any branch longer than its `--max-tokens` limit instead of truncating it; such rejections
stay visible failures and are not retried.
"""

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx

from s1mb.data import InferenceCase, ModelInfo, Prediction, Question, check_probabilities

from .base import questions_for_api, score_options

TRANSIENT = {429, 500, 502, 503, 504}
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 256
MAX_SCORE_LEVELS = 10


def transport_value(value):
    """Keep strings, objects, arrays and null; send other JSON scalars as their JSON text."""
    if value is None or isinstance(value, (str, dict, list)):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class GimbalAdapter:
    """Typed-question client for a running `gimbal_serve` server."""

    def __init__(
        self,
        model: str,
        revision: str,
        *,
        base_url: str = "http://127.0.0.1:8000",
        served_model: str | None = None,
        timeout: float = 600,
        case_batch_size: int = 1,
        retries: int = 2,
        runtime_settings: dict | None = None,
    ):
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP URL without credentials, query or fragment")
        if not model:
            raise ValueError("Specify the model identity")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Specify the exact 40-character checkpoint commit SHA as the revision")
        if served_model == "":
            raise ValueError("served_model must be nonempty when given")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive")
        for name, value, low, high in [
            ("case_batch_size", case_batch_size, 1, 32), ("retries", retries, 0, 5),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        settings = {} if runtime_settings is None else runtime_settings
        if not isinstance(settings, dict):
            raise TypeError("runtime_settings must be an object")
        self.runtime_settings = json.loads(json.dumps(settings, allow_nan=False))
        self.model, self.revision = model, revision
        self.base_url = str(url).rstrip("/") + "/"
        self.served_model = served_model
        self.resolved: str | None = None
        self.case_batch_size = case_batch_size
        self.timeout, self.retries = timeout, retries
        self._lock = Lock()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def requests_for(self, case: InferenceCase) -> list[tuple[dict, list[tuple[Question, list[str]]]]]:
        """Request bodies plus, per transport question, the declared option IDs in transport order."""
        if not case.questions:
            raise ValueError("A case requires at least one question")
        if len({q.id for q in case.questions}) != len(case.questions):
            raise ValueError("Duplicate question IDs")
        if case.images:
            raise ValueError("The Gimbal-27B server accepts text inputs only")
        for q in case.questions:
            if q.task not in {"choice", "noul", "score"}:
                raise ValueError("The Gimbal-27B server supports Choice, Noul and Score only")
            if q.task == "choice" and len(q.options) > MAX_CHOICE_OPTIONS:
                raise ValueError("Candidate count exceeds the server's 256-option capacity")
            if q.task == "score" and len(q.options) > MAX_SCORE_LEVELS:
                raise ValueError("The server supports at most ten Score levels")
        state = case.state
        if not isinstance(state, (str, dict)):
            state = json.dumps(state, ensure_ascii=False, allow_nan=False)
        requests = []
        for start in range(0, len(case.questions), MAX_QUESTIONS):
            chunk = case.questions[start:start + MAX_QUESTIONS]
            transport = [
                q.model_copy(update={"id": f"question_{i}", "task": "choice" if q.task == "noul" else q.task})
                for i, q in enumerate(chunk)
            ]
            questions = questions_for_api(transport, sort_score=True, structured=True, anonymous_choice=True)
            for body in questions.values():
                body["instructions"] = transport_value(body["instructions"])
                criteria = body["criteria"]
                body["criteria"] = (
                    [transport_value(v) for v in criteria] if isinstance(criteria, list)
                    else {k: transport_value(v) for k, v in criteria.items()}
                )
            order = [
                (q, [o.id for o in (score_options(q, True) if q.task == "score" else q.options)])
                for q in chunk
            ]
            requests.append(({"state": state, "questions": questions}, order))
        return requests

    def predict(self, case: InferenceCase) -> list[Prediction]:
        predictions = []
        for body, order in self.requests_for(case):
            if self.served_model:
                body = {**body, "model": self.served_model}
            payload = self._post(json.dumps(body, ensure_ascii=False, allow_nan=False).encode())
            predictions.extend(self._decode(case, payload, order))
        return predictions

    def _post(self, content: bytes) -> dict:
        response = None
        for attempt in range(self.retries + 1):
            with self._lock:
                self.usage["requests"] += 1
                self.usage["retries"] += int(attempt > 0)
            try:
                response = self.client.post("v1/systemone", content=content,
                                            headers={"Content-Type": "application/json"})
            except httpx.TransportError:
                if attempt == self.retries:
                    # Transport exceptions can contain URLs; keep the saved failure generic.
                    raise RuntimeError("Gimbal transport failure") from None
                time.sleep(2**attempt)
                continue
            if response.status_code not in TRANSIENT or attempt == self.retries:
                break
            time.sleep(2**attempt)
        if response is None:
            raise RuntimeError("Gimbal transport failure")
        if response.is_error:
            # Response bodies can echo inputs; record only the status.
            raise RuntimeError(f"Gimbal HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Invalid server response")
        return payload

    def _decode(self, case: InferenceCase, payload: dict, order) -> list[Prediction]:
        resolved = payload.get("model")
        if not isinstance(resolved, str) or not resolved:
            raise ValueError("Missing served model identity")
        if self.served_model is not None and resolved != self.served_model:
            raise ValueError("Served model differs from the configured name")
        tokens = payload.get("usage")
        if not isinstance(tokens, dict) or any(
            isinstance(tokens.get(k), bool) or not isinstance(tokens.get(k), int) or tokens[k] < 0
            for k in ("input_tokens", "output_tokens")
        ):
            raise ValueError("Invalid server token usage")
        answers = payload.get("answers")
        if not isinstance(answers, dict) or set(answers) != {f"question_{i}" for i in range(len(order))}:
            raise ValueError("Returned question IDs differ from the request")
        result = []
        for i, (q, ids) in enumerate(order):
            answer = answers[f"question_{i}"]
            if not isinstance(answer, dict) or answer.get("type") != ("score" if q.task == "score" else "choice"):
                raise ValueError("Returned task differs from the request")
            raw = answer.get("raw_probabilities")
            keys = ([str(j) for j in range(len(ids))] if q.task == "score"
                    else [f"option_{j}" for j in range(len(ids))])
            if not isinstance(raw, dict) or set(raw) != set(keys):
                raise ValueError("Missing full-precision probabilities; start the server with --full-precision")
            if any(isinstance(raw[k], bool) or not isinstance(raw[k], (int, float)) for k in keys):
                raise ValueError("Invalid probability")
            probabilities = {option_id: float(raw[key]) for option_id, key in zip(ids, keys, strict=True)}
            check_probabilities(probabilities, [o.id for o in q.options])
            result.append(Prediction(case_id=case.case_id, question_id=q.id, probabilities=probabilities))
        with self._lock:
            if self.resolved is not None and self.resolved != resolved:
                raise ValueError("Served model changed during the run")
            self.resolved = resolved
            for k in ("input_tokens", "output_tokens"):
                self.usage[k] += tokens[k]
        return result

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction] | Exception]:
        def predict_one(case: InferenceCase) -> list[Prediction] | Exception:
            try:
                return self.predict(case)
            except Exception as exc:  # noqa: BLE001 - keep request failures per case
                return exc

        with ThreadPoolExecutor(max_workers=self.case_batch_size) as pool:
            return list(pool.map(predict_one, cases))

    def metadata(self) -> ModelInfo:
        return ModelInfo(id=self.model, adapter="gimbal", revision=self.revision, settings={
            "backend": "caller-managed-gimbal-serve-vllm", "base_url": self.base_url,
            "endpoint": "/v1/systemone", "served_model_name": self.resolved or self.served_model,
            "revision_source": "caller-declared-checkpoint-revision",
            "runtime": dict(self.runtime_settings), "case_batch_size": self.case_batch_size,
            "questions_per_request": MAX_QUESTIONS, "question_chunking": "consecutive-64-same-state-v1",
            "timeout_seconds": self.timeout, "max_retries": self.retries,
            "renderer": "server-native-premise-hypothesis", "choice_keys": "anonymous-v1",
            "noul_transport": "explicit-choice-v1", "state_transport": "string-or-object-else-json-text-v1",
            "value_transport": "string-object-array-null-else-json-text-v1",
            "score_criteria_order": "ascending-value-v1", "temperature": "server-calibration-json",
            "input_length_policy": "server-max-tokens-rejection-no-truncation",
            "probability_source": "raw_probabilities-full-precision",
            "probability_normalization": "none",
        })

    def close(self) -> None:
        """Close this HTTP client; the server keeps running."""
        self.client.close()
