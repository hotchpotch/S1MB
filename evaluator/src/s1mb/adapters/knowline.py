"""KnowLine (PelaAI) through its own `/v1/systemone` server.

The model repos ship `knowline_server.py` and `serve_knowline.sh`, which serve the TypeSafe v1 typed-question contract
locally (SGLang with FP8 at load, or transformers). This adapter reuses the TypeSafe request/response mapping with a
local base URL, no credentials and unrounded probabilities, and sends up to `case_batch_size` cases concurrently
(answers do not depend on concurrency; each question is one prefill).

    bash serve_knowline.sh PelaAI/KnowLine-4B-Gen2 0 8080      # in the model repo
    KNOWLINE_BASE_URL=http://127.0.0.1:8080 s1mb run --adapter knowline \
        --model PelaAI/KnowLine-4B-Gen2 --revision <commit> --category english-v1 --run-id <id>
"""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx

from s1mb.data import InferenceCase, ModelInfo, Prediction

from .typesafe import TypeSafeAdapter


class KnowLineAdapter(TypeSafeAdapter):
    provider = "knowline"
    base_url_env = "KNOWLINE_BASE_URL"
    rounding_decimals = 6
    max_questions = 64

    def __init__(self, model: str, revision: str | None = None, case_batch_size: int = 16):
        self._lock = Lock()
        self.model = model
        self.revision = revision
        self.case_batch_size = max(1, case_batch_size)
        self.resolved: str | None = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
        self.base_url = os.environ.get(self.base_url_env, "http://127.0.0.1:8080")
        self.client = httpx.Client(base_url=self.base_url, timeout=600)

    def predict(self, case: InferenceCase) -> list[Prediction]:
        """The server takes up to 64 questions per request (the Jev limit). Larger cases are sent in chunks of 64 with
        the same state; each question is scored on its own, so chunking does not change any answer."""
        if len(case.questions) <= self.max_questions:
            return super().predict(case)
        out: list[Prediction] = []
        for i in range(0, len(case.questions), self.max_questions):
            out += super().predict(
                case.model_copy(update={"questions": case.questions[i : i + self.max_questions]})
            )
        return out

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        with ThreadPoolExecutor(max_workers=self.case_batch_size) as pool:
            return list(pool.map(self.predict, cases))

    def metadata(self) -> ModelInfo:
        info = super().metadata()
        settings = dict(info.settings)
        settings.update(
            {
                "server": "knowline_server.py from the model repo, behind SGLang (FP8 at load), chat style, temperature 1",
                "served_model_name": self.resolved,
                "concurrent_cases": self.case_batch_size,
                "questions_per_request": self.max_questions,
                "probability_normalization": "server softmax over option labels; sum-normalized within 1e-6",
            }
        )
        return info.model_copy(
            update={"id": self.model, "revision": self.revision or self.model, "settings": settings}
        )
