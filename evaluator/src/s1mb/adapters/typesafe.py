"""TypeSafe HTTP integration using the documented v1 typed-question contract."""

import os
import time
from threading import Lock

import httpx

from s1mb.data import InferenceCase, ModelInfo, Prediction

from .base import decode_answers, questions_for_api


class TypeSafeAdapter:
    provider = "typesafe"
    api_key_env = "TYPESAFE_API_KEY"
    base_url = "https://api.typesafe.ai"
    endpoint = "/v1/systemone"
    rounding_decimals = 2

    def __init__(self, model: str):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"Set {self.api_key_env} to use the {self.provider} adapter")
        self._lock = Lock()
        self.model = model
        self.resolved: str | None = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=120,
            headers={"Authorization": f"Bearer {key}"},
        )

    def predict(self, case: InferenceCase) -> list[Prediction]:
        for attempt in range(3):
            with self._lock:
                self.usage["requests"] += 1
                self.usage["retries"] += int(attempt > 0)
            try:
                response = self.client.post(
                    self.endpoint,
                    json={
                        "model": self.model,
                        "state": case.state,
                        "questions": questions_for_api(
                            case.questions, sort_score=True, structured=True
                        ),
                    },
                )
            except httpx.TransportError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
                continue
            if response.status_code not in {429, 500, 502, 503, 504, 529} or attempt == 2:
                break
            time.sleep(2**attempt)
        if response.is_error:
            # Do not persist response bodies, which can contain inputs or credentials.
            raise RuntimeError(f"{self.provider} HTTP {response.status_code}")
        payload = response.json()
        with self._lock:
            if self.resolved is not None and self.resolved != payload["model"]:
                raise ValueError("Resolved model changed during the run")
            self.resolved = payload["model"]
            for k in ("input_tokens", "output_tokens"):
                self.usage[k] += payload["usage"][k]
        # Live Jev 1.13 responses use two decimal places; sums can be 0.99 or 1.01.
        return decode_answers(
            case,
            payload["answers"],
            rounded=True,
            rounding_decimals=self.rounding_decimals,
            sort_score=True,
        )

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            id=self.model,
            adapter=self.provider,
            revision=self.resolved or self.model,
            settings={
                "requested_model": self.model,
                "score_criteria_order": "ascending-value-v1",
                "renderer": "system-one-structured-v2",
                "input_length_policy": "full-input-no-local-truncation",
                "probability_normalization": (
                    f"sum-normalized-within-{self.rounding_decimals}-decimal-rounding"
                ),
            },
        )

    def close(self) -> None:
        self.client.close()
