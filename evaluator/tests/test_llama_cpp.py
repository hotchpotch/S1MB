"""Exercise caller-managed llama.cpp inference using synthetic HTTP responses."""

import json

import httpx
import pytest

from s1mb.adapters.llama_cpp import LlamaCppAdapter
from s1mb.data import InferenceCase, Question


def case():
    return InferenceCase(case_id="source-case", state={"text": ["A", 1]}, questions=[
        Question(id="source-choice", task="choice", instructions="Select", instructions_json='{"instruction":"Select"}',
                 system_prompt="Authored system", options=[
                     {"id": "source-z", "description": "Z", "description_json": '{"meaning":"Z"}'},
                     {"id": "source-a", "description": "A"}]),
        Question(id="source-noul", task="noul", instructions="Condition?", options=[
            {"id": "true", "description": "Authored yes"}, {"id": "false", "description": "Authored no"}]),
        Question(id="source-score", task="score", instructions="Rate", options=[
            {"id": "high", "description": "High", "value": 10},
            {"id": "low", "description": "Low", "value": -2},
            {"id": "mid", "description": "Mid", "value": 3}]),
    ])


def payload():
    return {"model": "served", "usage": {"input_tokens": 100, "output_tokens": 0}, "answers": {
        "question_0": {"type": "choice", "probabilities": {"option_0": 0.3, "option_1": 0.7}},
        "question_1": {"type": "noul", "noul": 0.8},
        "question_2": {"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
    }}


@pytest.fixture
def adapter():
    value = LlamaCppAdapter("organization/model", "checkpoint-sha", served_model="served")
    value.client.close()
    yield value
    value.close()


def mock(adapter, handler):
    adapter.client = httpx.Client(base_url=adapter.base_url, transport=httpx.MockTransport(handler))


def test_native_mapping_preserves_structures_and_numeric_level_order(adapter):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=payload())
    mock(adapter, handler)
    result = adapter.predict(case())
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/systemone"
    assert body["state"] == case().state
    assert body["questions"]["question_0"]["instructions"] == {
        "system": "Authored system", "instruction": {"instruction": "Select"}}
    assert body["questions"]["question_0"]["criteria"] == {"option_0": {"meaning": "Z"}, "option_1": "A"}
    assert body["questions"]["question_1"]["criteria"] == {"true": "Authored yes", "false": "Authored no"}
    assert body["questions"]["question_2"]["criteria"] == ["Low", "Mid", "High"]
    assert b"source-" not in seen[0].content
    assert result[0].probabilities == {"source-z": 0.3, "source-a": 0.7}
    assert result[1].probabilities == pytest.approx({"true": 0.8, "false": 0.2})
    assert result[2].probabilities == {"low": 0.1, "mid": 0.2, "high": 0.7}
    assert [r.question_id for r in result] == [q.id for q in case().questions]
    assert all(r.case_id == "source-case" for r in result)
    assert adapter.metadata().revision == "checkpoint-sha"
    assert adapter.metadata().settings["served_model_name"] == "served"
    assert adapter.usage == {"input_tokens": 100, "output_tokens": 0, "requests": 1, "retries": 0}


@pytest.mark.parametrize("problem", ["keys", "nan", "sum", "task", "model", "truncated", "usage"])
def test_rejects_malformed_native_responses(adapter, problem):
    value = payload()
    if problem == "keys":
        value["answers"]["question_0"]["probabilities"] = {"other": 1.0}
    elif problem == "nan":
        value["answers"]["question_1"]["noul"] = "nan"
    elif problem == "sum":
        value["answers"]["question_0"]["probabilities"]["option_0"] = 0.2
    elif problem == "task":
        value["answers"]["question_0"]["type"] = "score"
    elif problem == "model":
        value["model"] = "unexpected"
    elif problem == "truncated":
        value["truncated"] = True
    else:
        value["usage"]["input_tokens"] = -1
    mock(adapter, lambda _: httpx.Response(200, json=value))
    with pytest.raises(ValueError):
        adapter.predict(case())


def test_rejects_model_drift_without_explicit_alias(adapter):
    adapter.served_model = None
    value = payload()
    mock(adapter, lambda _: httpx.Response(200, json=value))
    adapter.predict(case())
    value["model"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        adapter.predict(case())


@pytest.mark.parametrize("limit", ["questions", "candidates", "bytes"])
def test_rejects_local_capacity_overflow_before_any_call(adapter, limit):
    if limit == "questions":
        adapter.max_questions = 1
    elif limit == "candidates":
        adapter.max_candidates = 2
    else:
        adapter.max_request_bytes = 1
    mock(adapter, lambda _: pytest.fail("Overflow must be rejected before sending"))
    with pytest.raises(ValueError):
        adapter.predict(case())


@pytest.mark.parametrize("status", [400, 413])
def test_server_overflow_and_errors_are_not_retried_or_fabricated(adapter, status):
    mock(adapter, lambda _: httpx.Response(status, text="private input and credential"))
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as exc:
        adapter.predict(case())
    assert "private" not in str(exc.value)
    assert adapter.usage["requests"] == 1


def test_retry_is_bounded_and_usage_counts_attempts(adapter, monkeypatch):
    monkeypatch.setattr("s1mb.adapters.llama_cpp.time.sleep", lambda _: None)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json=payload())
    mock(adapter, handler)
    adapter.predict(case())
    assert adapter.usage["requests"] == 2
    assert adapter.usage["retries"] == 1
    assert adapter.usage["input_tokens"] == 100


def test_batch_preserves_case_alignment_and_individual_failures(adapter):
    adapter.case_batch_size = 2
    def handler(request):
        if json.loads(request.content)["state"] == "fail":
            return httpx.Response(413)
        return httpx.Response(200, json=payload())
    mock(adapter, handler)
    cases = [case(), case().model_copy(update={"state": "fail", "case_id": "failed"}),
             case().model_copy(update={"case_id": "third"})]
    results = adapter.predict_batch(cases)
    assert isinstance(results[1], RuntimeError)
    assert isinstance(results[0], list) and results[0][0].case_id == "source-case"
    assert isinstance(results[2], list) and results[2][0].case_id == "third"


def test_api_key_and_client_cleanup_leave_server_external(monkeypatch):
    monkeypatch.setenv("S1MB_LLAMA_API_KEY", "test-secret")
    adapter = LlamaCppAdapter("model", "sha", base_url="http://localhost:8080/prefix")
    assert adapter.client.headers["Authorization"] == "Bearer test-secret"
    assert "test-secret" not in adapter.metadata().model_dump_json()
    adapter.close()
    assert adapter.client.is_closed


@pytest.mark.parametrize("kwargs", [
    {"base_url": "http://user:secret@localhost"}, {"base_url": "http://localhost?key=secret"},
    {"case_batch_size": 0}, {"retries": -1}, {"retries": True}, {"timeout": float('nan')},
    {"runtime_settings": []}, {"temperature": 1},
])
def test_invalid_kwargs_fail_without_contacting_a_server(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LlamaCppAdapter("model", "sha", **kwargs)


def test_proxy_prefix_and_transport_retry_exhaustion(adapter, monkeypatch):
    monkeypatch.setattr("s1mb.adapters.llama_cpp.time.sleep", lambda _: None)
    adapter.base_url = "http://localhost:8080/proxy/"
    paths = []
    def handler(request):
        paths.append(request.url.path)
        raise httpx.ReadTimeout("secret response", request=request)
    mock(adapter, handler)
    with pytest.raises(RuntimeError, match="transport failure") as exc:
        adapter.predict(case())
    assert "secret" not in str(exc.value)
    assert paths == ["/proxy/v1/systemone"] * 3
    assert adapter.usage["requests"] == 3
    assert adapter.usage["retries"] == 2


def test_cli_passes_kwargs_and_closes_client(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from s1mb import cli
    from s1mb.data import Benchmark, Category

    selected = []
    closed = []
    class FakeAdapter:
        def __init__(self, model, revision, **kwargs):
            selected.append((model, revision, kwargs))
        def close(self):
            closed.append(True)
    benchmark = Benchmark(id="synthetic-choice", task="choice", dataset="unused", split="test",
                          case_count=1, decision_count=1, primary_metric="target_mass_at_prediction")
    monkeypatch.setattr("s1mb.dataset_source.dataset_session", lambda *a, **k: nullcontext())
    monkeypatch.setattr(cli, "load_category", lambda *a: Category(id="synthetic", name="Synthetic",
                       description="Test", benchmarks=[benchmark.id]))
    monkeypatch.setattr(cli, "load_benchmark", lambda *a: benchmark)
    monkeypatch.setattr("s1mb.adapters.llama_cpp.LlamaCppAdapter", FakeAdapter)
    monkeypatch.setattr("s1mb.runner.evaluate", lambda *a: SimpleNamespace(counts=SimpleNamespace(failed=0),
                       status="complete", metrics={}, elapsed_seconds=0, environment={}))
    monkeypatch.setattr("sys.argv", ["s1mb", "run", "--adapter", "llama-cpp", "--model", "model",
                       "--revision", "sha", "--adapter-kwargs", '{"base_url":"http://localhost:8000","timeout":30}',
                       "--output", str(tmp_path), "--run-id", "synthetic-cli", "--offline-dataset"])
    cli.main()
    assert selected == [("model", "sha", {"base_url": "http://localhost:8000", "timeout": 30})]
    assert closed == [True]
