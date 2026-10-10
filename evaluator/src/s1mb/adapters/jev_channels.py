"""JEV project hosted channels sharing the /v1/systemone typed-question contract.

FastinoGlideAdapter and OpenRouterAdapter are plain HTTP integrations like
typesafe.py; ClefCFAdapter shells out to the Cloudflare `cf` CLI, whose
Workers AI inference responses use the same answer schema. Providers may add
extra fields (confidence on noul, expected_level on score); decode_answers
ignores them. Batch inference runs bounded concurrent requests because the
runner loop itself is serial.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx

from s1mb.data import InferenceCase, ModelInfo, Prediction

from .base import decode_answers, questions_for_api

RETRY_STATUS = {429, 500, 502, 503, 504, 529}
SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


class SystemOneHTTPAdapter:
    """Common /v1/systemone HTTP client: retries, usage, concurrency."""

    case_batch_size = 8

    def __init__(
        self,
        *,
        provider: str,
        api_key_env: str,
        base_url: str,
        headers: dict,
        model: str,
        concurrency: int = 6,
        timeout: float = 120.0,
        endpoint: str = "/v1/systemone",
        rounding_decimals: int = 4,
        state_transform=None,
        question_chunk: int = 24,
    ):
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"Set {api_key_env} to use the {provider} adapter")
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.provider = provider
        self.model = model
        self.endpoint = endpoint
        self.concurrency = concurrency
        self.state_transform = state_transform
        self.question_chunk = question_chunk
        self.rounding_decimals = rounding_decimals
        self._lock = Lock()
        self.resolved: str | None = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
        self.client = httpx.Client(
            base_url=base_url, timeout=timeout, headers=headers
        )

    def _post(self, body: dict):
        for attempt in range(3):
            with self._lock:
                self.usage["requests"] += 1
                self.usage["retries"] += int(attempt > 0)
            try:
                response = self.client.post(self.endpoint, json=body)
            except httpx.TransportError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
                continue
            if response.status_code not in RETRY_STATUS or attempt == 2:
                break
            time.sleep(2**attempt)
        if response.is_error:
            # Do not persist response bodies, which can contain inputs or credentials;
            # a provider error message is safe and makes failures diagnosable.
            detail = ""
            try:
                detail = response.json().get("error", {}).get("message", "")[:200]
            except Exception:  # noqa: BLE001, S110 - detail stays empty on non-JSON errors
                pass
            raise RuntimeError(f"{self.provider} HTTP {response.status_code}: {detail}")
        payload = response.json()
        with self._lock:
            if self.resolved is not None and self.resolved != payload["model"]:
                raise ValueError("Resolved model changed during the run")
            self.resolved = payload["model"]
            for k in ("input_tokens", "output_tokens"):
                self.usage[k] += payload["usage"][k]
        return payload["answers"]

    def _request_answers(self, case: InferenceCase) -> dict:
        payload_questions = questions_for_api(
            case.questions, sort_score=True, structured=True
        )
        state = self.state_transform(case.state) if self.state_transform else case.state
        # Some providers (Fastino GLiDE) require instructions to be a plain
        # string; structured instruction objects are sent as serialized JSON text.
        if getattr(self, "coerce_string_instructions", False):
            payload_questions = {
                k: {**v, "instructions": v["instructions"] if isinstance(v["instructions"], str)
                    else json.dumps(v["instructions"], ensure_ascii=False)}
                for k, v in payload_questions.items()
            }
        # Some providers enforce ^[A-Za-z0-9_.-]+$ on question keys; authored ids
        # like "BLOCK DESTINATION" are transported as anonymous keys and mapped back.
        transport = payload_questions
        back = None
        if any(not SAFE_KEY.match(k) for k in payload_questions):
            keys = list(payload_questions)
            back = {f"q_{i:06d}": k for i, k in enumerate(keys)}
            transport = {alias: payload_questions[k] for alias, k in back.items()}
        if len(transport) <= self.question_chunk:
            answers = self._post(
                {"model": self.model, "state": state, "questions": transport}
            )
            return {back[a]: v for a, v in answers.items()} if back else answers
        # Providers cap questions per request (e.g. 422 on 54-question cases);
        # chunk with the same state. Questions are scored independently, so
        # chunking does not change any answer (same approach as the KnowLine adapter).
        merged: dict = {}
        keys = list(transport)
        for start in range(0, len(keys), self.question_chunk):
            chunk = {k: transport[k] for k in keys[start:start + self.question_chunk]}
            merged.update(self._post({"model": self.model, "state": state, "questions": chunk}))
        return {back[a]: v for a, v in merged.items()} if back else merged

    def predict(self, case: InferenceCase) -> list[Prediction]:
        answers = self._request_answers(case)
        return decode_answers(
            case,
            answers,
            rounded=True,
            rounding_decimals=self.rounding_decimals,
            sort_score=True,
        )

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(cases))) as pool:
            outputs = list(pool.map(self.predict, cases))
        return outputs

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
                "concurrency": self.concurrency,
            },
        )

    def close(self) -> None:
        self.client.close()


class FastinoGlideAdapter(SystemOneHTTPAdapter):
    def __init__(self, model: str = "fastino/glide", concurrency: int = 6):
        super().__init__(
            provider="fastino",
            api_key_env="FASTINO_API_KEY",
            base_url="https://api.fastino.ai",
            headers={},  # auth header set below: X-API-Key, Bearer also accepted
            model=model,
            concurrency=concurrency,
        )
        self.client.headers["X-API-Key"] = os.environ["FASTINO_API_KEY"]
        self.coerce_string_instructions = True


class OpenRouterAdapter(SystemOneHTTPAdapter):
    def __init__(self, model: str, concurrency: int = 6):
        if not model:
            raise ValueError("openrouter adapter requires --model")
        super().__init__(
            provider="openrouter",
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai",
            endpoint="/api/v1/systemone",
            headers={},
            model=model,
            concurrency=concurrency,
        )
        self.client.headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"


class SpanAdapter(OpenRouterAdapter):
    """span-01-lite restricts state to a string; send JSON-serialized state."""

    def __init__(self, model: str = "respan/span-01-lite:free", concurrency: int = 6):
        super().__init__(model, concurrency=concurrency)
        self.state_transform = lambda state: json.dumps(state, ensure_ascii=False)


class ClefCFAdapter:
    """Cloudflare Workers AI clef via the authenticated `cf` CLI subprocess."""

    provider = "clef-cf"
    case_batch_size = 6

    def __init__(self, model: str = "@cf/cloudflare/clef-flash", concurrency: int = 4):
        if shutil.which("cf") is None:
            raise ValueError("cf CLI not found on PATH; run `cf auth login` first")
        if not model.startswith("@cf/cloudflare/"):
            raise ValueError("model must be a @cf/cloudflare/clef[-flash] id")
        self.model = model
        self.concurrency = concurrency
        self._lock = Lock()
        self.resolved: str | None = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}

    def _payload(self, case: InferenceCase) -> dict:
        return {
            "state": case.state,
            "questions": questions_for_api(
                case.questions, sort_score=True, structured=True
            ),
        }

    def _request_answers(self, case: InferenceCase) -> dict:
        body = self._payload(case)
        for attempt in range(3):
            with self._lock:
                self.usage["requests"] += 1
                self.usage["retries"] += int(attempt > 0)
            fd, path = tempfile.mkstemp(suffix=".json", prefix="s1mb-clef-")
            try:
                with os.fdopen(fd, "w") as handle:
                    json.dump(body, handle)
                completed = subprocess.run(
                    ["cf", "ai", "run", self.model, "--quiet", "--body", f"@{path}"],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=180,
                )
            finally:
                os.unlink(path)
            if completed.returncode == 0:
                break
            # The CLI prints transient AiError 5xx/429 bodies to stdout/stderr.
            if attempt == 2 or "5007" in completed.stdout + completed.stderr:
                raise RuntimeError(
                    f"{self.provider} CLI exit {completed.returncode}: "
                    f"{(completed.stderr or completed.stdout)[-200:]!r}"
                )
            time.sleep(2**attempt)
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{self.provider} returned non-JSON output") from exc
        with self._lock:
            if self.resolved is not None and self.resolved != payload["model"]:
                raise ValueError("Resolved model changed during the run")
            self.resolved = payload["model"]
            for k in ("input_tokens", "output_tokens"):
                self.usage[k] += payload["usage"][k]
        return payload["answers"]

    def predict(self, case: InferenceCase) -> list[Prediction]:
        answers = self._request_answers(case)
        return decode_answers(
            case, answers, rounded=True, rounding_decimals=4, sort_score=True
        )

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(cases))) as pool:
            outputs = list(pool.map(self.predict, cases))
        return outputs

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            id=self.model,
            adapter=self.provider,
            revision=self.resolved or self.model,
            settings={
                "requested_model": self.model,
                "transport": "cf-cli-subprocess",
                "score_criteria_order": "ascending-value-v1",
                "renderer": "system-one-structured-v2",
                "input_length_policy": "full-input-no-local-truncation",
                "probability_normalization": "sum-normalized-within-4-decimal-rounding",
                "concurrency": self.concurrency,
            },
        )

    def close(self) -> None:
        pass
