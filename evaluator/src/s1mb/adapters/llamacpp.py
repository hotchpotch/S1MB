"""Native decision GGUFs through llama.cpp's own `/v1/systemone` endpoint.

`llama-server` serves a GGUF that carries decision metadata (`<arch>.decision.type`, for example `pplx-decider`) on the
TypeSafe v1 typed-question contract (llama.cpp `tools/server/README.md`, POST `/v1/systemone`). This adapter reuses the
TypeSafe request/response mapping with a local base URL, no credentials and unrounded probabilities, and sends up to
`case_batch_size` cases concurrently (each question is one prefill, so concurrency does not change any answer). The
server's build and loaded model file are read from `/props` and recorded; pass the GGUF repo commit as `--revision`
and, optionally, the file's sha256 in `LLAMACPP_GGUF_SHA256`.

    llama-server -m Whittle-Reflex-25B-A3B-Q8_0.gguf -ngl 99 -fa on -c 131072 -np 16 --kv-unified --port 8080
    LLAMACPP_BASE_URL=http://127.0.0.1:8080 s1mb run --adapter llamacpp \
        --model logic65/Whittle-Reflex-25B-A3B-GGUF --revision <commit> --category english-v1 --run-id <id>
"""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx

from s1mb.data import InferenceCase, ModelInfo, Prediction

from .typesafe import TypeSafeAdapter


class LlamaCppAdapter(TypeSafeAdapter):
    provider = "llamacpp"
    base_url_env = "LLAMACPP_BASE_URL"
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
        self.props: dict | None = None

    def predict(self, case: InferenceCase) -> list[Prediction]:
        """Cases above 64 questions go in chunks of 64 with the same state; questions are answered independently."""
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

    def server_props(self) -> dict:
        if self.props is None:
            try:
                response = self.client.get("/props")
                response.raise_for_status()
                self.props = response.json()
            except (httpx.HTTPError, ValueError):
                self.props = {}
        return self.props

    def metadata(self) -> ModelInfo:
        info = super().metadata()
        props = self.server_props()
        defaults = props.get("default_generation_settings") or {}
        settings = dict(info.settings)
        settings.update(
            {
                "server": "llama.cpp llama-server, POST /v1/systemone",
                "llama_cpp_build": props.get("build_info"),
                "model_file": os.path.basename(props.get("model_path") or "") or None,
                "model_file_sha256": os.environ.get("LLAMACPP_GGUF_SHA256") or None,
                "served_model_name": self.resolved,
                "server_context_per_slot": defaults.get("n_ctx"),
                "server_slots": props.get("total_slots"),
                "concurrent_cases": self.case_batch_size,
                "questions_per_request": self.max_questions,
                "probability_normalization": "server softmax over option labels; sum-normalized within 1e-6",
            }
        )
        return info.model_copy(
            update={"id": self.model, "revision": self.revision or self.model, "settings": settings}
        )
