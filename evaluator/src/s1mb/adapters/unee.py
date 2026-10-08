"""Unee decision models through their self-hosted, Jev-compatible server.

`unee serve` (pip install unee) starts llama.cpp with a Unee GGUF file and exposes /v1/systemone, which accepts
the documented typed-question requests unchanged. So this is the TypeSafe contract with a local base URL, no API
key and four-decimal probabilities. Models: https://huggingface.co/uneeverse

    pip install unee huggingface_hub
    hf download uneeverse/unee-0.8b-GGUF unee-0.8b-Q4_K_M.gguf --local-dir .
    unee serve --model unee-0.8b-Q4_K_M.gguf --model-name unee-0.8b-Q4_K_M --gpu-layers 99 --port 8000
    uv run s1mb run --adapter unee --model unee-0.8b

`unee serve` needs llama.cpp's llama-server on PATH (or --llama-server). Set UNEE_BASE_URL when the server is
not on http://127.0.0.1:8000. Long-document benchmarks need a slot that fits their inputs, for example
`--slots 1 --ctx 32768`; the server rejects longer inputs instead of truncating them.
"""

import os
from threading import Lock

import httpx

from s1mb.data import ModelInfo

from .typesafe import TypeSafeAdapter


class UneeAdapter(TypeSafeAdapter):
    provider = "unee"
    base_url = "http://127.0.0.1:8000"
    rounding_decimals = 4

    def __init__(self, model: str):
        self.base_url = os.environ.get("UNEE_BASE_URL", self.base_url)
        self._lock = Lock()
        self.model = model
        self.resolved: str | None = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
        self.client = httpx.Client(base_url=self.base_url, timeout=120)

    def metadata(self) -> ModelInfo:
        model = super().metadata()
        model.settings["endpoint"] = self.base_url + self.endpoint
        return model
