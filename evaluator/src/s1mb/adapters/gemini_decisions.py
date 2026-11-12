"""Gemini Decisions API with anonymous transport IDs and native probabilities."""

import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx

from s1mb.data import InferenceCase, ModelInfo, Prediction, Question, check_probabilities

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_BYTES = 8 * 1024 * 1024
BASE_URL = "https://generativelanguage.googleapis.com"
QUESTION_KEY = "question"
# Content-filter reasons from the SDK's BlockReason; the API reports a blocked
# input either as HTTP 200 prompt feedback or as HTTP 400 with this error code.
BLOCK_REASONS = {"safety", "other", "blocklist", "prohibited_content", "image_safety"}


class InputBlocked(Exception):
    """The provider's content filter rejected the input."""


def blocked_error(data: bytes) -> bool:
    try:
        error = json.loads(data).get("error")
    except (ValueError, AttributeError):
        return False
    return isinstance(error, dict) and error.get("code") in BLOCK_REASONS


def request_question(question: Question) -> dict:
    """Preserve authored descriptions; transport names never expose benchmark IDs."""
    instructions = question.instructions_json or question.instructions
    if question.system_prompt:
        instructions = question.system_prompt + "\n\n" + instructions
    descriptions = [o.description_json or o.description for o in question.options]
    if question.task == "noul":
        by_id = dict(zip([o.id for o in question.options], descriptions, strict=True))
        return {
            "type": "noul",
            "instructions": instructions,
            "criteria": {
                "true_description": by_id["true"],
                "false_description": by_id["false"],
            },
        }
    if question.task == "choice":
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": {f"option_{i}": d for i, d in enumerate(descriptions)},
        }
    if question.task == "score":
        # API levels are 0-based indices; keep the authored numeric value visible.
        return {
            "type": "score",
            "instructions": instructions,
            "criteria": [
                f"{option.value:g}: {description}"
                for option, description in zip(question.options, descriptions, strict=True)
            ],
        }
    raise ValueError("Gemini Decisions does not support Box questions")


def decision_input(case: InferenceCase) -> tuple[str | list[dict], list[dict]]:
    """Send each image once with a neutral label; retain authored state verbatim."""
    state = (
        case.state_json
        if case.state_json is not None
        else json.dumps(case.state, ensure_ascii=False)
    )
    if not case.images:
        return state, []
    from s1mb.vision import prepare_image

    content = [{"type": "text", "text": state}]
    receipts = []
    for name, asset in case.images.items():
        payload, mime, receipt = prepare_image(asset)
        content.append({"type": "text", "text": name})
        content.append(
            {
                "type": "image",
                "data": base64.b64encode(payload).decode("ascii"),
                "mime_type": mime,
                "resolution": "high",
            }
        )
        receipts.append(receipt)
    return content, receipts


def decode_answer(case: InferenceCase, question: Question, answer: dict) -> Prediction:
    """Use per-level probabilities, never the API's expected score."""
    if not isinstance(answer, dict):
        raise TypeError("Expected an answer object")
    if answer.get("type") != question.task:
        raise ValueError("Returned task differs")
    if question.task == "noul":
        raw = answer["noul"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("Invalid noul probability")
        probabilities = {"false": 1 - raw, "true": raw}
    else:
        rows = answer["probabilities"]
        keys = [
            f"option_{i}" if question.task == "choice" else str(i)
            for i in range(len(question.options))
        ]
        if not isinstance(rows, dict) or set(rows) != set(keys):
            raise ValueError("Returned candidate values differ")
        for value in rows.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Invalid candidate probability")
        probabilities = {o.id: rows[key] for o, key in zip(question.options, keys, strict=True)}
    check_probabilities(probabilities, [o.id for o in question.options])
    return Prediction(case_id=case.case_id, question_id=question.id, probabilities=probabilities)


class GeminiDecisionsAdapter:
    """Bounded HTTP inference against the fixed official endpoint."""

    supports_images = True

    def __init__(self, model: str = "gemini-decision-1", case_batch_size: int = 4):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY to use gemini-decisions")
        if not model or "/" in model or not 1 <= case_batch_size <= 32:
            raise ValueError("A bare model ID and case batch size within 1..32 are required")
        self.model = model
        self.case_batch_size = case_batch_size
        self.resolved: str | None = None
        self.image_receipts: dict[str, list[dict]] = {}
        self._lock = Lock()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
        self.client = httpx.Client(
            base_url=BASE_URL,
            timeout=120,
            headers={"x-goog-api-key": key},
            follow_redirects=False,
        )

    def _request(self, method: str, path: str, content: bytes | None = None) -> dict:
        for attempt in range(3):
            with self._lock:
                self.usage["requests"] += 1
                self.usage["retries"] += int(attempt > 0)
            try:
                with self.client.stream(
                    method, path, content=content, headers={"Content-Type": "application/json"}
                ) as response:
                    status = response.status_code
                    if status == 200:
                        data = bytearray()
                        for chunk in response.iter_bytes():
                            if len(data) + len(chunk) > MAX_BYTES:
                                raise ValueError("Gemini Decisions response exceeds byte limit")
                            data.extend(chunk)
                        payload = json.loads(data)
                        break
                    if status == 400:
                        data = response.read()
                        if len(data) <= MAX_BYTES and blocked_error(data):
                            raise InputBlocked
            except httpx.TransportError:
                if attempt == 2:
                    raise RuntimeError("Gemini Decisions transport failure") from None
                time.sleep(2**attempt)
                continue
            if status not in RETRY_STATUS or attempt == 2:
                raise RuntimeError(f"Gemini Decisions HTTP {status}")
            time.sleep(2**attempt)
        if not isinstance(payload, dict):
            raise TypeError("Expected a response object")
        return payload

    def _resolve(self) -> None:
        """Record the served model version once; the decision response omits it."""
        with self._lock:
            if self.resolved is not None:
                return
        payload = self._request("GET", f"/v1beta/models/{self.model}")
        version = payload.get("version")
        if payload.get("name") != f"models/{self.model}" or not isinstance(version, str):
            raise ValueError("Gemini model metadata is invalid")
        if "createDecision" not in (payload.get("supportedGenerationMethods") or []):
            raise ValueError("Gemini model does not support createDecision")
        with self._lock:
            self.resolved = version

    def _post(self, body: dict) -> dict:
        content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        if len(content) > MAX_BYTES:
            raise ValueError("Gemini Decisions request exceeds byte limit")
        payload = self._request("POST", "/v1beta/decisions", content)
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            raise TypeError("Expected usage object")
        with self._lock:
            for field, name in (
                ("total_input_tokens", "input_tokens"),
                ("total_output_tokens", "output_tokens"),
            ):
                value = usage.get(field, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError("Invalid API usage count")
                self.usage[name] += value
        return payload

    def predict(self, case: InferenceCase) -> list[Prediction]:
        state, receipts = decision_input(case)
        if receipts:
            with self._lock:
                self.image_receipts[case.case_id] = receipts
        predictions = []
        for question in case.questions:
            try:
                body = {
                    "model": f"models/{self.model}",
                    "input": state,
                    "questions": {QUESTION_KEY: request_question(question)},
                }
                self._resolve()
                payload = self._post(body)
                feedback = payload.get("prompt_feedback") or {}
                if not isinstance(feedback, dict):
                    raise TypeError("Expected prompt feedback object")
                if feedback.get("block_reason"):
                    raise InputBlocked
                answers = payload.get("answers")
                if not isinstance(answers, dict) or set(answers) != {QUESTION_KEY}:
                    raise ValueError("Gemini Decisions returned an invalid answer set")
                predictions.append(decode_answer(case, question, answers[QUESTION_KEY]))
            except InputBlocked:
                predictions.append(
                    Prediction(case_id=case.case_id, question_id=question.id, abstention="refusal")
                )
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                # Never save upstream bodies or malformed values in result errors.
                error = (
                    str(exc)
                    if isinstance(exc, RuntimeError)
                    else "Gemini Decisions invalid response or input"
                )
                predictions.append(
                    Prediction(case_id=case.case_id, question_id=question.id, error=error)
                )
        return predictions

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        with ThreadPoolExecutor(max_workers=self.case_batch_size) as pool:
            return list(pool.map(self.predict, cases))

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            id=self.model,
            adapter="gemini-decisions",
            revision=self.resolved or self.model,
            settings={
                "requested_model": self.model,
                "endpoint": f"{BASE_URL}/v1beta/decisions",
                "renderer": "authored-text-anonymous-options-numeric-levels-v1",
                "questions_per_request": 1,
                "case_batch_size": self.case_batch_size,
                "input_length_policy": "full-input-no-local-truncation",
                "score_criteria_order": "declared-order-v1",
                "probability_normalization": "none",
                "noul_probability_range": [0.01, 0.99],
                "block_policy": "content-filter-block-is-refusal-v2",
                "vision_transport": "named-inline-image-v1",
                "image_resolution": "high",
                "image_preprocessing": "shared-full-image-policy-v1",
                "box_support": False,
            },
        )

    def close(self) -> None:
        self.client.close()
