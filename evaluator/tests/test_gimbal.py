"""Exercise the caller-managed Gimbal-27B server adapter with synthetic HTTP responses."""

import json

import httpx
import pytest

from s1mb.adapters.gimbal import GimbalAdapter
from s1mb.data import InferenceCase, Question

SHA = "0123456789abcdef0123456789abcdef01234567"


def case(state=None):
    return InferenceCase(case_id="source-case", state={"text": ["A", 1]} if state is None else state, questions=[
        Question(id="source-choice", task="choice", instructions="Select", instructions_json='{"instruction":"Select"}',
                 options=[{"id": "source-z", "description": "Z", "description_json": '{"meaning":"Z"}'},
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
        "question_0": {"type": "choice", "choice": "option_1", "probabilities": {"option_0": 0.3, "option_1": 0.7},
                       "raw_probabilities": {"option_0": 0.304, "option_1": 0.696}},
        "question_1": {"type": "choice", "choice": "option_0", "probabilities": {"option_0": 0.8, "option_1": 0.2},
                       "raw_probabilities": {"option_0": 0.8125, "option_1": 0.1875}},
        "question_2": {"type": "score", "score": 1.6, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                       "raw_probabilities": {"0": 0.125, "1": 0.25, "2": 0.625}},
    }}


@pytest.fixture
def adapter():
    value = GimbalAdapter("organization/model", SHA, served_model="served")
    value.client.close()
    yield value
    value.close()


def mock(adapter, handler):
    adapter.client = httpx.Client(base_url=adapter.base_url, transport=httpx.MockTransport(handler))


def test_transport_preserves_meanings_and_reads_full_precision(adapter):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=payload())
    mock(adapter, handler)
    result = adapter.predict(case())
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/systemone"
    assert body["state"] == case().state
    assert body["model"] == "served"
    assert body["questions"]["question_0"] == {
        "type": "choice", "instructions": {"instruction": "Select"},
        "criteria": {"option_0": {"meaning": "Z"}, "option_1": "A"}}
    assert body["questions"]["question_1"] == {
        "type": "choice", "instructions": "Condition?",
        "criteria": {"option_0": "Authored yes", "option_1": "Authored no"}}
    assert body["questions"]["question_2"] == {
        "type": "score", "instructions": "Rate", "criteria": ["Low", "Mid", "High"]}
    assert b"source-" not in seen[0].content
    assert result[0].probabilities == {"source-z": 0.304, "source-a": 0.696}
    assert result[1].probabilities == {"true": 0.8125, "false": 0.1875}
    assert result[2].probabilities == {"low": 0.125, "mid": 0.25, "high": 0.625}
    assert [r.question_id for r in result] == [q.id for q in case().questions]
    assert all(r.case_id == "source-case" for r in result)
    info = adapter.metadata()
    assert info.revision == SHA and info.adapter == "gimbal"
    assert info.settings["served_model_name"] == "served"
    assert info.settings["noul_transport"] == "explicit-choice-v1"
    assert info.settings["probability_source"] == "raw_probabilities-full-precision"
    assert adapter.usage == {"input_tokens": 100, "output_tokens": 0, "requests": 1, "retries": 0}


def test_noul_definitions_keep_declared_order():
    value = case()
    value.questions[1] = value.questions[1].model_copy(update={"options": list(reversed(value.questions[1].options))})
    body, order = GimbalAdapter("organization/model", SHA).requests_for(value)[0]
    assert body["questions"]["question_1"]["criteria"] == {"option_0": "Authored no", "option_1": "Authored yes"}
    assert order[1][1] == ["false", "true"]


def test_json_scalars_are_sent_as_their_json_text():
    value = case()
    value.questions[0] = value.questions[0].model_copy(update={"options": [
        value.questions[0].options[0].model_copy(update={"description_json": "7.5"}),
        value.questions[0].options[1].model_copy(update={"description_json": "true"})]})
    value.questions[2] = value.questions[2].model_copy(update={"instructions_json": "3"})
    body, _ = GimbalAdapter("organization/model", SHA).requests_for(value)[0]
    assert body["questions"]["question_0"]["criteria"] == {"option_0": "7.5", "option_1": "true"}
    assert body["questions"]["question_2"]["instructions"] == "3"


@pytest.mark.parametrize("state, sent", [
    (["first", {"k": "é"}], '["first", {"k": "é"}]'), (12, "12"), ("plain text", "plain text"), ({"k": 1}, {"k": 1})])
def test_states_other_than_strings_and_objects_are_sent_as_json_text(state, sent):
    body, _ = GimbalAdapter("organization/model", SHA).requests_for(case(state))[0]
    assert body["state"] == sent


def test_cases_above_64_questions_are_chunked_with_the_same_state(adapter):
    questions = [Question(id=f"source-{i}", task="choice", instructions=f"Q{i}", options=[
        {"id": "a", "description": "A"}, {"id": "b", "description": "B"}]) for i in range(65)]
    value = InferenceCase(case_id="wide", state={"text": "S"}, questions=questions)
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"model": "served", "usage": {"input_tokens": 1, "output_tokens": 0},
                                         "answers": {k: {"type": "choice", "raw_probabilities": {
                                             "option_0": 0.25, "option_1": 0.75}} for k in body["questions"]}})
    mock(adapter, handler)
    result = adapter.predict(value)
    assert [len(b["questions"]) for b in bodies] == [64, 1]
    assert all(b["state"] == {"text": "S"} for b in bodies)
    assert bodies[1]["questions"]["question_0"]["instructions"] == "Q64"
    assert [r.question_id for r in result] == [q.id for q in questions]
    assert result[64].probabilities == {"a": 0.25, "b": 0.75}
    assert adapter.usage["requests"] == 2 and adapter.usage["input_tokens"] == 2


@pytest.mark.parametrize("problem", ["rounded-only", "keys", "sum", "bool", "task", "answer", "model", "usage",
                                     "ids", "payload"])
def test_rejects_malformed_responses(adapter, problem):
    value = payload()
    answer = value["answers"]["question_0"]
    if problem == "rounded-only":
        del answer["raw_probabilities"]
    elif problem == "keys":
        answer["raw_probabilities"] = {"other": 1.0}
    elif problem == "sum":
        answer["raw_probabilities"]["option_0"] = 0.2
    elif problem == "bool":
        answer["raw_probabilities"] = {"option_0": False, "option_1": True}
    elif problem == "task":
        answer["type"] = "noul"
    elif problem == "answer":
        value["answers"]["question_0"] = ["choice"]
    elif problem == "model":
        value["model"] = "unexpected"
    elif problem == "usage":
        value["usage"]["input_tokens"] = -1
    elif problem == "ids":
        value["answers"]["question_9"] = value["answers"].pop("question_2")
    else:
        value = [value]
    mock(adapter, lambda _: httpx.Response(200, json=value))
    with pytest.raises(TypeError if problem == "payload" else ValueError):
        adapter.predict(case())


def test_rejects_model_drift_without_a_configured_name(adapter):
    adapter.served_model = None
    value = payload()
    mock(adapter, lambda _: httpx.Response(200, json=value))
    adapter.predict(case())
    value["model"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        adapter.predict(case())


@pytest.mark.parametrize("problem", ["choice", "score", "image", "box", "duplicate"])
def test_rejects_unsupported_inputs_before_any_call(adapter, problem):
    value = case()
    if problem == "choice":
        options = [{"id": f"o{i}", "description": f"O{i}"} for i in range(257)]
        value.questions[0] = value.questions[0].model_copy(update={"options": options})
    elif problem == "score":
        options = [{"id": f"l{i}", "description": f"L{i}", "value": i} for i in range(11)]
        value.questions[2] = value.questions[2].model_copy(update={"options": options})
    elif problem == "image":
        value = value.model_copy(update={"images": {"image": {"path": "a.png", "sha256": "0" * 64}}})
    elif problem == "box":
        value.questions[0] = Question(id="source-box", task="box", instructions="Find")
    else:
        value.questions[1] = value.questions[1].model_copy(update={"id": "source-choice"})
    mock(adapter, lambda _: pytest.fail("Unsupported input must be rejected before sending"))
    with pytest.raises(ValueError):
        adapter.predict(value)


@pytest.mark.parametrize("status", [400, 413])
def test_server_rejections_are_not_retried_or_fabricated(adapter, status):
    mock(adapter, lambda _: httpx.Response(status, text="private input"))
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as exc:
        adapter.predict(case())
    assert "private" not in str(exc.value)
    assert adapter.usage["requests"] == 1


def test_transient_errors_are_retried_with_a_bound(adapter, monkeypatch):
    monkeypatch.setattr("s1mb.adapters.gimbal.time.sleep", lambda _: None)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json=payload())
    mock(adapter, handler)
    adapter.predict(case())
    assert adapter.usage["requests"] == 2 and adapter.usage["retries"] == 1
    mock(adapter, lambda _: httpx.Response(500))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        adapter.predict(case())
    assert adapter.usage["requests"] == 5
    def failing(request):
        raise httpx.ReadTimeout("secret url", request=request)
    mock(adapter, failing)
    with pytest.raises(RuntimeError, match="transport failure") as exc:
        adapter.predict(case())
    assert "secret" not in str(exc.value)


def test_batch_preserves_case_alignment_and_individual_failures(adapter):
    adapter.case_batch_size = 2
    def handler(request):
        if json.loads(request.content)["state"] == "fail":
            return httpx.Response(400)
        return httpx.Response(200, json=payload())
    mock(adapter, handler)
    cases = [case(), case("fail").model_copy(update={"case_id": "failed"}),
             case().model_copy(update={"case_id": "third"})]
    results = adapter.predict_batch(cases)
    assert isinstance(results[1], RuntimeError)
    assert not isinstance(results[0], Exception) and results[0][0].case_id == "source-case"
    assert not isinstance(results[2], Exception) and results[2][0].case_id == "third"


@pytest.mark.parametrize("kwargs", [
    {"revision": "main"}, {"base_url": "http://user:secret@localhost"}, {"base_url": "http://localhost?k=v"},
    {"case_batch_size": 0}, {"case_batch_size": 33}, {"retries": True}, {"retries": 6}, {"timeout": float("nan")},
    {"served_model": ""}, {"runtime_settings": []},
])
def test_invalid_settings_fail_without_contacting_a_server(kwargs):
    revision = kwargs.pop("revision", SHA)
    with pytest.raises((ValueError, TypeError)):
        GimbalAdapter("model", revision, **kwargs)


def test_cli_passes_kwargs_and_rejects_foreign_ones(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from s1mb import cli
    from s1mb.data import Benchmark, Category

    selected, closed = [], []
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
    monkeypatch.setattr("s1mb.adapters.gimbal.GimbalAdapter", FakeAdapter)
    monkeypatch.setattr("s1mb.runner.evaluate", lambda *a: SimpleNamespace(counts=SimpleNamespace(failed=0),
                       status="complete", metrics={}, elapsed_seconds=0, environment={}))
    argv = ["s1mb", "run", "--adapter", "gimbal", "--model", "model", "--revision", SHA, "--output", str(tmp_path),
            "--offline-dataset", "--adapter-kwargs"]
    monkeypatch.setattr("sys.argv", argv + ['{"base_url":"http://localhost:8100","case_batch_size":16}',
                                            "--run-id", "synthetic-gimbal"])
    cli.main()
    assert selected == [("model", SHA, {"base_url": "http://localhost:8100", "case_batch_size": 16})]
    assert closed == [True]
    monkeypatch.setattr("sys.argv", argv + ['{"max_candidates":52}', "--run-id", "synthetic-gimbal-2"])
    with pytest.raises(SystemExit):
        cli.main()
