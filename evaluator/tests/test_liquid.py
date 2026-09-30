"""Check Liquid routing and provenance without real credentials or API calls."""

import json

import httpx
import pytest
from test_typesafe import case

from s1mb.adapters.liquid import LiquidAdapter


def test_liquid_endpoint_and_metadata(monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid-test-only")
    adapter = LiquidAdapter("d1:free")
    adapter.client.close()

    def handler(request):
        assert str(request.url) == "https://api.liquid.ai/decisions/v1/systemone"
        payload = json.loads(request.content)
        assert payload["model"] == "d1:free"
        assert payload["questions"]["q"]["criteria"] == {"false": "No", "true": "Yes"}
        return httpx.Response(200, json={
            "model": "d1-resolved",
            "answers": {"q": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
        })

    adapter.client = httpx.Client(
        base_url=adapter.base_url, transport=httpx.MockTransport(handler)
    )
    try:
        probabilities = adapter.predict(case())[0].probabilities
        assert probabilities is not None
        assert probabilities["true"] == 0.9
        assert adapter.metadata().adapter == "liquid"
        assert adapter.metadata().revision == "d1-resolved"
        assert adapter.usage["output_tokens"] == 0
        assert "liquid-test-only" not in adapter.metadata().model_dump_json()
    finally:
        adapter.close()


def test_liquid_does_not_use_typesafe_credentials(monkeypatch):
    monkeypatch.delenv("LIQUID_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    with pytest.raises(ValueError, match="LIQUID_API_KEY"):
        LiquidAdapter("d1:free")


def test_batch_preserves_order_and_isolates_failures(monkeypatch):
    from threading import Barrier

    monkeypatch.setenv("LIQUID_API_KEY", "test-only")
    adapter = LiquidAdapter("d1:free", case_batch_size=2)
    adapter.client.close()
    barrier = Barrier(2)

    def handler(request):
        barrier.wait(timeout=5)
        payload = json.loads(request.content)
        if payload["state"] == "bad":
            return httpx.Response(400)
        return httpx.Response(200, json={
            "model": "d1:free",
            "answers": {"q": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
        })

    adapter.client = httpx.Client(
        base_url=adapter.base_url, transport=httpx.MockTransport(handler)
    )
    good = case()
    bad = good.model_copy(update={"case_id": "bad", "state": "bad"})
    try:
        outputs = adapter.predict_batch([good, bad])
        assert isinstance(outputs[0], list)
        assert outputs[0][0].case_id == "one"
        assert isinstance(outputs[1], RuntimeError)
        assert adapter.usage == {
            "input_tokens": 12, "output_tokens": 0, "requests": 2, "retries": 0,
        }
        assert adapter.metadata().settings["case_batch_size"] == 2
    finally:
        adapter.close()
