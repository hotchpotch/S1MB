"""Check the Kettei adapter against a fake API: no credentials, network or model."""

import json

import httpx
import pytest
from test_typesafe import case

from s1mb.adapters.kettei import KetteiAdapter
from s1mb.data import InferenceCase, Question


def noul_response(model="kettei-fat", value=0.9):
    return httpx.Response(200, json={
        "model": model,
        "answers": {"q": {"type": "noul", "noul": value}},
        "usage": {"input_tokens": 12, "output_tokens": 0},
    })


def fake_api(adapter, handler):
    """Replace only the transport, keeping the adapter's base URL and headers."""
    headers = adapter.client.headers
    adapter.client.close()
    adapter.client = httpx.Client(
        base_url=adapter.base_url, headers=headers, transport=httpx.MockTransport(handler)
    )


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("SQWISH_API_KEY", "sqwish-test-only")


def three_tasks():
    return InferenceCase(
        case_id="case-7",
        state={"ticket": {"subject": "Refund", "lines": [1, 2]}},
        questions=[
            Question(
                id="route",
                task="choice",
                instructions="Which team should handle this?",
                options=[
                    {"id": "billing", "description": "Billing"},
                    {"id": "abuse", "description": "Abuse", "description_json": '{"team": "abuse"}'},
                    {"id": "accounts", "description": "Accounts"},
                ],
            ),
            Question(
                id="refund",
                task="noul",
                instructions="Is a refund requested?",
                instructions_json='{"task": "refund", "strict": true}',
                options=[
                    {"id": "false", "description": "No refund is asked for"},
                    {"id": "true", "description": "The customer asks for money back"},
                ],
            ),
            Question(
                id="urgency",
                task="score",
                instructions="How urgent is it?",
                options=[
                    {"id": "high", "value": 3, "description": "Act now"},
                    {"id": "none", "value": 0, "description": "No action"},
                    {"id": "some", "value": 1, "description": "This week"},
                ],
            ),
        ],
    )


def test_request_shape_and_mapping_for_all_tasks(key):
    adapter = KetteiAdapter("kettei-fat")
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={
            "model": "kettei-fat",
            "answers": {
                "route": {"type": "choice", "probabilities": {
                    "billing": 0.7123456789, "abuse": 0.0876543211, "accounts": 0.2}},
                "refund": {"type": "noul", "noul": 0.8125},
                "urgency": {"type": "score", "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}},
            },
            "usage": {"input_tokens": 40, "output_tokens": 0},
        })

    fake_api(adapter, handler)
    try:
        predictions = {p.question_id: p.probabilities for p in adapter.predict(three_tasks())}
    finally:
        adapter.close()
    request = sent[0]
    assert str(request.url) == "https://api.sqwish.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer sqwish-test-only"
    payload = json.loads(request.content)
    assert payload["model"] == "kettei-fat"
    assert payload["state"] == {"ticket": {"subject": "Refund", "lines": [1, 2]}}
    questions = payload["questions"]
    assert list(questions["route"]["criteria"]) == ["billing", "abuse", "accounts"]
    assert questions["route"]["criteria"]["abuse"] == {"team": "abuse"}
    assert questions["refund"]["instructions"] == {"task": "refund", "strict": True}
    assert questions["refund"]["criteria"] == {
        "false": "No refund is asked for", "true": "The customer asks for money back"}
    assert questions["urgency"]["criteria"] == ["No action", "This week", "Act now"]
    assert b"case-7" not in request.content
    assert predictions["route"] == pytest.approx(
        {"billing": 0.7123456789, "abuse": 0.0876543211, "accounts": 0.2})
    assert predictions["refund"] == pytest.approx({"false": 0.1875, "true": 0.8125})
    assert predictions["urgency"] == pytest.approx({"none": 0.1, "some": 0.3, "high": 0.6})


def test_metadata_records_route_and_settings_without_the_key(key):
    adapter = KetteiAdapter("kettei-chub", case_batch_size=4)
    fake_api(adapter, lambda request: noul_response("kettei-chub"))
    try:
        adapter.predict(case())
        metadata = adapter.metadata()
    finally:
        adapter.close()
    assert metadata.adapter == "kettei"
    assert metadata.revision == "kettei-chub"
    assert metadata.settings == {
        "requested_model": "kettei-chub",
        "score_criteria_order": "ascending-value-v1",
        "renderer": "system-one-structured-v2",
        "input_length_policy": "full-input-no-local-truncation",
        "probability_normalization": "sum-normalized-within-4-decimal-rounding",
        "case_batch_size": 4,
        "endpoint": "https://api.sqwish.ai/v1/systemone",
    }
    assert "sqwish-test-only" not in metadata.model_dump_json()


def test_needs_its_own_key_and_never_uses_typesafe(monkeypatch):
    monkeypatch.delenv("SQWISH_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-test-only")
    with pytest.raises(ValueError, match="SQWISH_API_KEY"):
        KetteiAdapter("kettei-fat")
    monkeypatch.setenv("SQWISH_API_KEY", "sqwish-test-only")
    adapter = KetteiAdapter("kettei-fat")
    try:
        assert adapter.client.base_url.host == "api.sqwish.ai"
        assert adapter.client.headers["authorization"] == "Bearer sqwish-test-only"
    finally:
        adapter.close()


@pytest.mark.parametrize("size", [0, 33])
def test_case_batch_size_is_bounded(key, size):
    with pytest.raises(ValueError, match="between 1 and 32"):
        KetteiAdapter("kettei-fat", case_batch_size=size)


@pytest.mark.parametrize("answers", [
    {"q": {"type": "choice", "probabilities": {"false": 0.1, "true": 0.9}}},
    {"other": {"type": "noul", "noul": 0.9}},
])
def test_rejects_mismatched_answers(key, answers):
    adapter = KetteiAdapter("kettei-fat")
    fake_api(adapter, lambda request: httpx.Response(200, json={
        "model": "kettei-fat", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}))
    try:
        with pytest.raises(ValueError):
            adapter.predict(case())
    finally:
        adapter.close()


def test_rejects_probabilities_that_do_not_sum_to_one(key):
    c = three_tasks()
    c.questions = c.questions[:1]
    adapter = KetteiAdapter("kettei-fat")
    fake_api(adapter, lambda request: httpx.Response(200, json={
        "model": "kettei-fat",
        "answers": {"route": {"type": "choice", "probabilities": {
            "billing": 0.6, "abuse": 0.1, "accounts": 0.2}}},
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }))
    try:
        with pytest.raises(ValueError, match="rounding"):
            adapter.predict(c)
    finally:
        adapter.close()


def test_rejected_input_is_not_retried_and_keeps_only_the_status(key):
    adapter = KetteiAdapter("kettei-fat")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(422, json={"error": {"message": "input too long: A report"}})

    fake_api(adapter, handler)
    try:
        with pytest.raises(RuntimeError) as failure:
            adapter.predict(case())
    finally:
        adapter.close()
    assert str(failure.value) == "kettei HTTP 422"
    assert len(calls) == 1
    assert adapter.usage["retries"] == 0


def test_transient_status_is_retried_and_counted(key, monkeypatch):
    monkeypatch.setattr("s1mb.adapters.typesafe.time.sleep", lambda _: None)
    adapter = KetteiAdapter("kettei-fat")
    replies = iter([httpx.Response(503), noul_response()])
    fake_api(adapter, lambda request: next(replies))
    try:
        assert adapter.predict(case())[0].probabilities == pytest.approx({"false": 0.1, "true": 0.9})
    finally:
        adapter.close()
    assert adapter.usage == {"input_tokens": 12, "output_tokens": 0, "requests": 2, "retries": 1}


def test_batch_preserves_order_and_isolates_failures(key):
    from threading import Barrier

    adapter = KetteiAdapter("kettei-fat", case_batch_size=2)
    barrier = Barrier(2)

    def handler(request):
        barrier.wait(timeout=5)
        if json.loads(request.content)["state"] == "bad":
            return httpx.Response(400)
        return noul_response()

    fake_api(adapter, handler)
    good = case()
    bad = good.model_copy(update={"case_id": "bad", "state": "bad"})
    try:
        outputs = adapter.predict_batch([good, bad])
    finally:
        adapter.close()
    assert isinstance(outputs[0], list)
    assert outputs[0][0].case_id == "one"
    assert isinstance(outputs[1], RuntimeError)
    assert adapter.usage == {"input_tokens": 12, "output_tokens": 0, "requests": 2, "retries": 0}


def test_cli_builds_the_adapter_with_case_batch_size(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from s1mb import cli
    from s1mb.data import Benchmark, Category

    built = []
    closed = []

    class FakeAdapter:
        def __init__(self, model, case_batch_size):
            built.append((model, case_batch_size))

        def close(self):
            closed.append(True)

    benchmark = Benchmark(id="synthetic-noul", task="noul", dataset="unused", split="test",
                          case_count=1, decision_count=1, primary_metric="binary_brier")
    monkeypatch.setattr("s1mb.dataset_source.dataset_session", lambda *a, **k: nullcontext())
    monkeypatch.setattr(cli, "load_category", lambda *a: Category(
        id="synthetic", name="Synthetic", description="Test", benchmarks=[benchmark.id]))
    monkeypatch.setattr(cli, "load_benchmark", lambda *a: benchmark)
    monkeypatch.setattr("s1mb.adapters.kettei.KetteiAdapter", FakeAdapter)
    monkeypatch.setattr("s1mb.runner.evaluate", lambda *a: SimpleNamespace(
        counts=SimpleNamespace(failed=0), status="complete", metrics={}, elapsed_seconds=0,
        environment={}))
    monkeypatch.setattr("sys.argv", [
        "s1mb", "run", "--adapter", "kettei", "--model", "kettei-chub", "--case-batch-size", "4",
        "--output", str(tmp_path), "--run-id", "synthetic-cli", "--offline-dataset"])
    cli.main()
    assert built == [("kettei-chub", 4)]
    assert closed == [True]
