"""Exercise the HTTP adapter without a network call or credentials."""

import httpx
import pytest

from s1mb.adapters.typesafe import TypeSafeAdapter
from s1mb.data import InferenceCase, Question


def case():
    return InferenceCase(
        case_id="one",
        state={"text": "A report"},
        questions=[
            Question(
                id="q",
                task="noul",
                instructions="Does the condition hold?",
                options=[
                    {"id": "false", "description": "No"},
                    {"id": "true", "description": "Yes"},
                ],
            )
        ],
    )


def test_api_maps_response_and_preserves_resolved_model(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    adapter = TypeSafeAdapter("jev-pinned")
    adapter.client.close()
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "jev-resolved",
                "answers": {"q": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    adapter.client = httpx.Client(
        base_url="https://api.typesafe.ai", transport=httpx.MockTransport(handler)
    )
    try:
        prediction = adapter.predict(case())[0]
        assert prediction.probabilities == pytest.approx({"false": 0.2, "true": 0.8})
        assert adapter.metadata().revision == "jev-resolved"
        assert "test-only" not in adapter.metadata().model_dump_json()
        assert seen[0].url.path == "/v1/systemone"
        assert b"target" not in seen[0].content
    finally:
        adapter.close()


def test_api_rejects_model_drift(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    adapter = TypeSafeAdapter("alias")
    adapter.client.close()
    adapter.resolved = "original"
    adapter.client = httpx.Client(
        base_url="https://api.typesafe.ai",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"model": "changed", "answers": {}})
        ),
    )
    try:
        with pytest.raises(ValueError, match="changed"):
            adapter.predict(case())
    finally:
        adapter.close()


@pytest.mark.parametrize("transient", ["overload", "timeout"])
@pytest.mark.dataset
def test_smoke_usage_is_per_benchmark_and_counts_retry(monkeypatch, tmp_path, transient):
    import json

    from s1mb.data import DATA_DIR, load_benchmark, load_category
    from s1mb.runner import evaluate

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    monkeypatch.setattr("s1mb.adapters.typesafe.time.sleep", lambda _: None)
    adapter = TypeSafeAdapter("jev-1.13.0")
    adapter.client.close()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            if transient == "timeout":
                raise httpx.ReadTimeout("Test timeout", request=request)
            return httpx.Response(529)
        answers = {}
        for key, q in json.loads(request.content)["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.5}
            else:
                ids = (
                    list(q["criteria"])
                    if q["type"] == "choice"
                    else [str(i) for i in range(len(q["criteria"]))]
                )
                answers[key] = {"type": q["type"], "probabilities": {k: 1 / len(ids) for k in ids}}
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        )

    adapter.client = httpx.Client(
        base_url="https://api.typesafe.ai", transport=httpx.MockTransport(handler)
    )
    try:
        for i, name in enumerate(load_category(DATA_DIR, "smoke-v1").benchmarks):
            result = evaluate(
                adapter, DATA_DIR, load_benchmark(DATA_DIR, name), tmp_path, "smoke", limit=3
            )
            assert result.counts.failed == 0
            assert result.environment["api_usage"] == {
                "input_tokens": 300,
                "output_tokens": 30,
                "requests": 4 if i == 0 else 3,
                "retries": 1 if i == 0 else 0,
            }
            assert "usage" not in result.model.settings
    finally:
        adapter.close()


def test_live_two_decimal_rounding_is_bounded():
    from s1mb.adapters.base import decode_answers

    c = case()
    c.questions[0].task = "choice"
    answers = {"q": {"type": "choice", "probabilities": {"false": 0.23, "true": 0.76}}}
    result = decode_answers(c, answers, rounded=True, rounding_decimals=2)
    assert result[0].probabilities == pytest.approx({"false": 0.23 / 0.99, "true": 0.76 / 0.99})
    with pytest.raises(ValueError, match="rounding"):
        decode_answers(c, answers, rounded=True)
    answers["q"]["probabilities"] = {"false": 0.2, "true": 0.7}
    with pytest.raises(ValueError, match="rounding"):
        decode_answers(c, answers, rounded=True, rounding_decimals=2)


def test_score_sorts_numeric_values_and_maps_response_to_original_ids(monkeypatch):
    import json

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    c = InferenceCase(
        case_id="score",
        state="Evidence",
        questions=[
            Question(
                id="q",
                task="score",
                instructions="Rate relevance",
                options=[
                    {"id": "high", "value": 4, "description": "Direct answer"},
                    {"id": "low", "value": 0, "description": "Not useful"},
                    {"id": "middle", "value": 1, "description": "Some context"},
                ],
            )
        ],
    )
    original = c.model_dump()
    adapter = TypeSafeAdapter("jev-pinned")
    adapter.client.close()

    def handler(request):
        assert json.loads(request.content)["questions"]["q"]["criteria"] == [
            "Not useful",
            "Some context",
            "Direct answer",
        ]
        return httpx.Response(
            200,
            json={
                "model": "jev-pinned",
                "usage": {"input_tokens": 10, "output_tokens": 1},
                "answers": {
                    "q": {"type": "score", "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}}
                },
            },
        )

    adapter.client = httpx.Client(
        base_url="https://api.typesafe.ai", transport=httpx.MockTransport(handler)
    )
    try:
        result = adapter.predict(c)[0]
        assert result.probabilities == {"low": 0.1, "middle": 0.3, "high": 0.6}
        assert c.model_dump() == original
        assert adapter.metadata().settings["score_criteria_order"] == "ascending-value-v1"
    finally:
        adapter.close()
