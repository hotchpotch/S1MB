"""Exercise the llama.cpp `/v1/systemone` adapter without a network call or a model server."""

import json

import httpx
import pytest

from s1mb.adapters.llamacpp import LlamaCppAdapter
from s1mb.data import InferenceCase, Question


def noul(i):
    return Question(
        id=f"q{i}",
        task="noul",
        instructions="Does the condition hold?",
        options=[{"id": "false", "description": "No"}, {"id": "true", "description": "Yes"}],
    )


def case(n=1):
    return InferenceCase(
        case_id="one", state={"text": "A report"}, questions=[noul(i) for i in range(n)]
    )


def server(seen):
    def handler(request):
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "build_info": "b11455-abeada33",
                    "model_path": "/models/Example-Q8_0.gguf",
                    "total_slots": 16,
                    "default_generation_settings": {"n_ctx": 8192},
                },
            )
        body = json.loads(request.content)
        seen.append((request, body))
        answers = {k: {"type": "noul", "noul": 0.8} for k in body["questions"]}
        return httpx.Response(
            200,
            json={
                "model": "Example-Q8_0.gguf",
                "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

    return handler


def adapter_with(handler):
    adapter = LlamaCppAdapter("example/Example-GGUF", "abc123")
    adapter.client.close()
    adapter.client = httpx.Client(
        base_url="http://127.0.0.1:8080", transport=httpx.MockTransport(handler)
    )
    return adapter


def test_maps_response_and_records_server(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("LLAMACPP_GGUF_SHA256", "f" * 64)
    seen = []
    adapter = adapter_with(server(seen))
    try:
        prediction = adapter.predict(case())[0]
        assert prediction.probabilities == pytest.approx({"false": 0.2, "true": 0.8})
        request, body = seen[0]
        assert request.url.path == "/v1/systemone"
        assert "authorization" not in request.headers
        assert body["model"] == "example/Example-GGUF"
        info = adapter.metadata()
        assert info.adapter == "llamacpp"
        assert info.revision == "abc123"
        assert info.settings["llama_cpp_build"] == "b11455-abeada33"
        assert info.settings["model_file"] == "Example-Q8_0.gguf"
        assert info.settings["model_file_sha256"] == "f" * 64
        assert info.settings["served_model_name"] == "Example-Q8_0.gguf"
        assert info.settings["server_context_per_slot"] == 8192
    finally:
        adapter.close()


def test_metadata_without_props():
    def handler(request):
        return httpx.Response(404)

    adapter = adapter_with(handler)
    try:
        info = adapter.metadata()
        assert info.settings["llama_cpp_build"] is None
        assert info.settings["model_file"] is None
    finally:
        adapter.close()


def test_splits_cases_over_the_question_limit():
    seen = []
    adapter = adapter_with(server(seen))
    try:
        predictions = adapter.predict(case(65))
        assert [len(body["questions"]) for _, body in seen] == [64, 1]
        assert [p.question_id for p in predictions] == [f"q{i}" for i in range(65)]
    finally:
        adapter.close()


def test_batches_cases_concurrently():
    seen = []
    adapter = adapter_with(server(seen))
    try:
        outputs = adapter.predict_batch([case(2), case(3)])
        assert [len(o) for o in outputs] == [2, 3]
        assert len(seen) == 2
    finally:
        adapter.close()
