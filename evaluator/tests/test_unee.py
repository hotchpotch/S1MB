"""Check Unee routing and provenance without a running server."""

import json

import httpx
from test_typesafe import case

from s1mb.adapters.unee import UneeAdapter


def test_unee_endpoint_metadata_and_no_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("UNEE_BASE_URL", "http://127.0.0.1:8123")
    adapter = UneeAdapter("unee-0.8b")
    adapter.client.close()

    def handler(request):
        assert str(request.url) == "http://127.0.0.1:8123/v1/systemone"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        assert payload["model"] == "unee-0.8b"
        assert payload["questions"]["q"]["criteria"] == {"false": "No", "true": "Yes"}
        return httpx.Response(200, json={
            "model": "unee-0.8b-Q4_K_M",
            "answers": {"q": {"type": "noul", "noul": 0.9123}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
        })

    adapter.client = httpx.Client(
        base_url=adapter.base_url, transport=httpx.MockTransport(handler)
    )
    try:
        probabilities = adapter.predict(case())[0].probabilities
        assert probabilities is not None
        assert probabilities["true"] == 0.9123
        metadata = adapter.metadata()
        assert metadata.adapter == "unee"
        assert metadata.revision == "unee-0.8b-Q4_K_M"
        assert metadata.settings["endpoint"] == "http://127.0.0.1:8123/v1/systemone"
    finally:
        adapter.close()


def test_unee_default_base_url(monkeypatch):
    monkeypatch.delenv("UNEE_BASE_URL", raising=False)
    adapter = UneeAdapter("unee-2b")
    try:
        assert adapter.base_url == "http://127.0.0.1:8000"
    finally:
        adapter.close()
