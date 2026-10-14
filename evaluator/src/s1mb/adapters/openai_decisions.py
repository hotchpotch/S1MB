"""OpenAI Decisions API with anonymous transport IDs and native probabilities."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx

from s1mb.data import InferenceCase, ModelInfo, Prediction, Question, check_probabilities

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_BYTES = 8 * 1024 * 1024


def request_question(question: Question) -> dict:
    """Preserve authored descriptions; transport names never expose benchmark IDs."""
    instructions = question.instructions_json or question.instructions
    if question.system_prompt:
        instructions = question.system_prompt + "\n\n" + instructions
    result = {"name": "question", "instructions": instructions}
    descriptions = [o.description_json or o.description for o in question.options]
    if question.task == "noul":
        criteria = {
            o.id: description for o, description in zip(question.options, descriptions, strict=True)
        }
        result.update(
            type="predicate",
            instructions=instructions
            + "\n\nReturn the probability of the true outcome under these authored criteria:\n"
            + json.dumps(criteria, ensure_ascii=False),
        )
    elif question.task == "choice":
        result.update(
            type="choice",
            choices=[
                {"value": f"option_{i}", "description": description}
                for i, description in enumerate(descriptions)
            ],
        )
    else:
        result.update(
            type="score",
            levels=[
                {"label": str(option.value), "description": description}
                for option, description in zip(question.options, descriptions, strict=True)
            ],
        )
    return result


def decode_answer(case: InferenceCase, question: Question, answer: dict) -> Prediction:
    """Use per-level probabilities, never the API's ordinal expected score."""
    if not isinstance(answer, dict):
        raise TypeError("Expected an answer object")
    if answer.get("name") != "question":
        raise ValueError("Returned question name differs")
    if answer.get("type") == "refusal":
        return Prediction(
            case_id=case.case_id, question_id=question.id, abstention="refusal"
        )
    expected = "predicate" if question.task == "noul" else question.task
    if answer.get("type") != expected:
        raise ValueError("Returned task differs")
    if question.task == "noul":
        raw = answer["probability"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("Invalid predicate probability")
        probabilities = {"false": 1 - raw, "true": raw}
    else:
        rows = answer["probabilities"]
        if not isinstance(rows, list) or len(rows) != len(question.options):
            raise ValueError("Returned candidate count differs")
        by_value = {}
        for row in rows:
            value, probability = row["value"], row["probability"]
            if isinstance(value, bool) or value in by_value:
                raise ValueError("Invalid or duplicate candidate")
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise TypeError("Invalid candidate probability")
            by_value[value] = probability
        values = [
            f"option_{i}" if question.task == "choice" else i for i in range(len(question.options))
        ]
        if set(by_value) != set(values):
            raise ValueError("Returned candidate values differ")
        probabilities = {
            o.id: by_value[value] for o, value in zip(question.options, values, strict=True)
        }
    check_probabilities(probabilities, [o.id for o in question.options])
    return Prediction(case_id=case.case_id, question_id=question.id, probabilities=probabilities)


class OpenAIDecisionsAdapter:
    """Bounded HTTP inference against the fixed official endpoint."""

    def __init__(self, model: str = "gpt-6-luna", case_batch_size: int = 4):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY to use openai-decisions")
        if not model or not 1 <= case_batch_size <= 32:
            raise ValueError("A model and case batch size within 1..32 are required")
        self.model = model
        self.case_batch_size = case_batch_size
        self.resolved: str | None = None
        self._lock = Lock()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
        self.client = httpx.Client(
            base_url="https://api.openai.com",
            timeout=120,
            headers={"Authorization": f"Bearer {key}"},
            follow_redirects=False,
        )

    def _post(self, body: dict) -> dict:
        content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        if len(content) > MAX_BYTES:
            raise ValueError("OpenAI Decisions request exceeds byte limit")
        for attempt in range(3):
            with self._lock:
                self.usage["requests"] += 1
                self.usage["retries"] += int(attempt > 0)
            try:
                with self.client.stream(
                    "POST",
                    "/v1/decisions",
                    content=content,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    status = response.status_code
                    if status == 200:
                        data = bytearray()
                        for chunk in response.iter_bytes():
                            if len(data) + len(chunk) > MAX_BYTES:
                                raise ValueError("OpenAI Decisions response exceeds byte limit")
                            data.extend(chunk)
                        payload = json.loads(data)
                        break
            except httpx.TransportError:
                if attempt == 2:
                    raise RuntimeError("OpenAI Decisions transport failure") from None
                time.sleep(2**attempt)
                continue
            if status not in RETRY_STATUS or attempt == 2:
                raise RuntimeError(f"OpenAI Decisions HTTP {status}")
            time.sleep(2**attempt)
        if not isinstance(payload, dict):
            raise TypeError("Expected a response object")
        resolved = payload.get("model")
        if not isinstance(resolved, str) or not resolved:
            raise ValueError("OpenAI Decisions response missing model")
        with self._lock:
            if self.resolved is not None and resolved != self.resolved:
                raise ValueError("Resolved OpenAI model changed during the run")
            self.resolved = resolved
            usage = payload.get("usage") or {}
            if not isinstance(usage, dict):
                raise TypeError("Expected usage object")
            for field in ("input_tokens", "output_tokens"):
                value = usage.get(field, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError("Invalid API usage count")
                self.usage[field] += value
        return payload

    def predict(self, case: InferenceCase) -> list[Prediction]:
        state = (
            case.state_json
            if case.state_json is not None
            else json.dumps(case.state, ensure_ascii=False)
        )
        predictions = []
        for question in case.questions:
            try:
                payload = self._post(
                    {"model": self.model, "input": state, "questions": [request_question(question)]}
                )
                answers = payload.get("answers")
                if not isinstance(answers, list) or len(answers) != 1:
                    raise ValueError("OpenAI Decisions returned an invalid answer count")
                predictions.append(decode_answer(case, question, answers[0]))
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                # Never save upstream bodies or malformed values in result errors.
                error = (
                    str(exc)
                    if isinstance(exc, RuntimeError)
                    else "OpenAI Decisions invalid response or input"
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
            adapter="openai-decisions",
            revision=self.resolved or self.model,
            settings={
                "requested_model": self.model,
                "endpoint": "https://api.openai.com/v1/decisions",
                "renderer": "authored-text-anonymous-options-numeric-levels-v2",
                "questions_per_request": 1,
                "case_batch_size": self.case_batch_size,
                "input_length_policy": "full-input-no-local-truncation",
                "score_criteria_order": "declared-order-v1",
                "probability_normalization": "none",
                "public_beta": True,
            },
        )

    def close(self) -> None:
        self.client.close()
