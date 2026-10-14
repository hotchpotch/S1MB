"""Offline tests for native task mapping and safe API failure handling."""

import json

import httpx
import pytest

from s1mb.adapters.openai_decisions import OpenAIDecisionsAdapter
from s1mb.data import InferenceCase, Option, Question


@pytest.mark.parametrize("task", ["choice", "noul", "score"])
def test_native_probabilities_and_anonymous_ids(monkeypatch, task):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ids = ["false", "true"] if task == "noul" else ["private-a", "private-b"]
    q = Question(
        id="private-question",
        task=task,
        instructions="Use authored criteria.",
        options=[
            Option(id=identifier, description=f"Criterion {i}", value=10 + i * 80)
            for i, identifier in enumerate(ids)
        ],
    )
    case = InferenceCase(
        case_id="private-case",
        state={"text": "hello"},
        state_json='{"text": "hello"}',
        questions=[q],
    )

    def respond(request):
        body = json.loads(request.content)
        assert body["input"] == case.state_json
        assert "private-" not in request.content.decode()
        question = body["questions"][0]
        assert "Use authored criteria." in question["instructions"]
        kind = "predicate" if task == "noul" else task
        answer = {"name": "question", "type": kind}
        if task == "noul":
            answer["probability"] = 0.7
            assert "Criterion 0" in question["instructions"]
        else:
            if task == "score":
                assert [level["label"] for level in question["levels"]] == ["10.0", "90.0"]
            values = ["option_0", "option_1"] if task == "choice" else [0, 1]
            answer["probabilities"] = [
                {"value": values[1], "probability": 0.7},
                {"value": values[0], "probability": 0.3},
            ]
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": [answer]})

    adapter = OpenAIDecisionsAdapter()
    adapter.client.close()
    adapter.client = httpx.Client(
        base_url="https://api.openai.com", transport=httpx.MockTransport(respond)
    )
    try:
        result = adapter.predict(case)[0]
        assert result.error is None
        assert result.probabilities == pytest.approx(dict(zip(ids, [0.3, 0.7], strict=True)))
        assert adapter.metadata().revision == "gpt-6-luna"
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "body,status",
    [
        ([], 200),
        ({"secret": "upstream text"}, 400),
        ({"error": {"message": "refusal"}}, 403),
        ({"model": "gpt-6-luna", "answers": [None]}, 200),
        (
            {
                "model": "gpt-6-luna",
                "answers": [{"name": "question", "type": "predicate", "probability": 1.1}],
            },
            200,
        ),
    ],
)
def test_failures_are_not_predictions_or_upstream_text(monkeypatch, body, status):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    q = Question(
        id="q",
        task="noul",
        instructions="Evaluate.",
        options=[Option(id="false", description="No"), Option(id="true", description="Yes")],
    )
    adapter = OpenAIDecisionsAdapter()
    adapter.client.close()
    adapter.client = httpx.Client(
        base_url="https://api.openai.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body)),
    )
    try:
        result = adapter.predict(InferenceCase(case_id="c", state="hello", questions=[q]))[0]
        assert result.error
        assert result.abstention is None
        assert result.probabilities is None
        assert "upstream text" not in result.error
    finally:
        adapter.close()


def test_retry_refusal_and_stable_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("s1mb.adapters.openai_decisions.time.sleep", lambda _: None)
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={
                "model": "gpt-6-luna",
                "answers": [
                    {"name": "question", "type": "refusal", "reason": "private upstream reason"}
                ],
                "usage": {"input_tokens": 20},
            },
        )

    q = Question(
        id="q",
        task="noul",
        instructions="Evaluate.",
        options=[Option(id="false", description="No"), Option(id="true", description="Yes")],
    )
    adapter = OpenAIDecisionsAdapter(case_batch_size=2)
    adapter.client.close()
    adapter.client = httpx.Client(
        base_url="https://api.openai.com", transport=httpx.MockTransport(respond)
    )
    before = adapter.metadata().settings
    try:
        cases = [InferenceCase(case_id=str(i), state="hello", questions=[q]) for i in range(2)]
        predictions = adapter.predict_batch(cases)
        assert [row[0].case_id for row in predictions] == ["0", "1"]
        assert all(row[0].abstention == "refusal" for row in predictions)
        assert adapter.usage["retries"] == 1
        assert adapter.usage["input_tokens"] == 40
        assert adapter.metadata().settings == before
    finally:
        adapter.close()


def test_request_bound_does_not_send(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("s1mb.adapters.openai_decisions.MAX_BYTES", 8)
    adapter = OpenAIDecisionsAdapter()
    try:
        with pytest.raises(ValueError, match="byte limit"):
            adapter._post({"input": "too large"})
        assert adapter.usage["requests"] == 0
    finally:
        adapter.close()


def test_refusal_preserves_other_answers_without_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        answer = (
            {"name": "question", "type": "refusal"}
            if calls == 1
            else {"name": "question", "type": "predicate", "probability": 0.7}
        )
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": [answer]})

    questions = [
        Question(
            id=qid,
            task="noul",
            instructions="Judge.",
            options=[Option(id="false", description="No"), Option(id="true", description="Yes")],
        )
        for qid in ("refused", "answered")
    ]
    adapter = OpenAIDecisionsAdapter()
    adapter.client.close()
    adapter.client = httpx.Client(
        base_url="https://api.openai.com", transport=httpx.MockTransport(respond)
    )
    try:
        results = adapter.predict(InferenceCase(case_id="c", state="hello", questions=questions))
        assert results[0].question_id == "refused"
        assert results[0].abstention == "refusal"
        assert results[0].probabilities is None and results[0].error is None
        assert results[1].question_id == "answered"
        assert results[1].probabilities == pytest.approx({"false": 0.3, "true": 0.7})
        assert results[1].abstention is None and results[1].error is None
        assert calls == 2
        assert adapter.usage["retries"] == 0
    finally:
        adapter.close()
