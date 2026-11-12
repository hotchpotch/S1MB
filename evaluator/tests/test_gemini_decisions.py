"""Offline tests for native task mapping and safe API failure handling."""

import json

import httpx
import pytest

from s1mb.adapters.gemini_decisions import BASE_URL, GeminiDecisionsAdapter
from s1mb.data import InferenceCase, Option, Question

MODEL = {
    "name": "models/gemini-decision-1",
    "version": "gemini-decision-1",
    "supportedGenerationMethods": ["createDecision"],
}


def mock(adapter, decide):
    def respond(request):
        if request.method == "GET":
            assert request.url.path == "/v1beta/models/gemini-decision-1"
            return httpx.Response(200, json=MODEL)
        assert request.url.path == "/v1beta/decisions"
        return decide(request)

    adapter.client.close()
    adapter.client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(respond))


def noul_question(qid="q"):
    return Question(
        id=qid,
        task="noul",
        instructions="Evaluate.",
        options=[Option(id="false", description="No"), Option(id="true", description="Yes")],
    )


@pytest.mark.parametrize("task", ["choice", "noul", "score"])
def test_native_probabilities_and_anonymous_ids(monkeypatch, task):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
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

    def decide(request):
        body = json.loads(request.content)
        assert body["model"] == "models/gemini-decision-1"
        assert body["input"] == case.state_json
        assert "private-" not in request.content.decode()
        question = body["questions"]["question"]
        assert question["type"] == task
        assert question["instructions"] == "Use authored criteria."
        answer = {"type": task, "confidence": 0.7}
        if task == "noul":
            assert question["criteria"] == {
                "true_description": "Criterion 1",
                "false_description": "Criterion 0",
            }
            answer["noul"] = 0.7
        elif task == "choice":
            assert question["criteria"] == {"option_0": "Criterion 0", "option_1": "Criterion 1"}
            answer["probabilities"] = {"option_1": 0.7, "option_0": 0.3}
        else:
            assert question["criteria"] == ["10: Criterion 0", "90: Criterion 1"]
            answer.update(probabilities={"1": 0.7, "0": 0.3}, score=0.7)
        return httpx.Response(200, json={"answers": {"question": answer}})

    adapter = GeminiDecisionsAdapter()
    mock(adapter, decide)
    try:
        result = adapter.predict(case)[0]
        assert result.error is None
        assert result.probabilities == pytest.approx(dict(zip(ids, [0.3, 0.7], strict=True)))
        assert adapter.metadata().revision == "gemini-decision-1"
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "body,status",
    [
        ([], 200),
        ({"secret": "upstream text"}, 400),
        ({"error": {"message": "upstream text"}}, 403),
        ({"answers": {"question": None}}, 200),
        ({"answers": {"other": {"type": "noul", "noul": 0.5}}}, 200),
        ({"answers": {"question": {"type": "noul", "noul": 1.1}}}, 200),
        ({"answers": {"question": {"type": "choice", "probabilities": {}}}}, 200),
    ],
)
def test_failures_are_not_predictions_or_upstream_text(monkeypatch, body, status):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    adapter = GeminiDecisionsAdapter()
    mock(adapter, lambda request: httpx.Response(status, json=body))
    try:
        result = adapter.predict(
            InferenceCase(case_id="c", state="hello", questions=[noul_question()])
        )[0]
        assert result.error
        assert result.abstention is None
        assert result.probabilities is None
        assert "upstream text" not in result.error
    finally:
        adapter.close()


def test_invalid_model_metadata_is_an_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    adapter = GeminiDecisionsAdapter()
    adapter.client.close()
    adapter.client = httpx.Client(
        base_url=BASE_URL,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={**MODEL, "supportedGenerationMethods": []})
        ),
    )
    try:
        result = adapter.predict(
            InferenceCase(case_id="c", state="hello", questions=[noul_question()])
        )[0]
        assert result.error and adapter.resolved is None
    finally:
        adapter.close()


def test_retry_block_and_stable_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("s1mb.adapters.gemini_decisions.time.sleep", lambda _: None)
    calls = 0

    def decide(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={
                "prompt_feedback": {
                    "block_reason": "safety",
                    "block_reason_message": "private upstream reason",
                },
                "usage": {"total_input_tokens": 20, "total_output_tokens": 0},
            },
        )

    adapter = GeminiDecisionsAdapter(case_batch_size=2)
    mock(adapter, decide)
    before = adapter.metadata().settings
    try:
        cases = [
            InferenceCase(case_id=str(i), state="hello", questions=[noul_question()])
            for i in range(2)
        ]
        predictions = adapter.predict_batch(cases)
        assert [row[0].case_id for row in predictions] == ["0", "1"]
        assert all(row[0].abstention == "refusal" for row in predictions)
        assert adapter.usage["retries"] == 1
        assert adapter.usage["input_tokens"] == 40
        assert adapter.metadata().settings == before
    finally:
        adapter.close()


def test_request_bound_does_not_send(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("s1mb.adapters.gemini_decisions.MAX_BYTES", 8)
    adapter = GeminiDecisionsAdapter()
    try:
        with pytest.raises(ValueError, match="byte limit"):
            adapter._post({"input": "too large"})
        assert adapter.usage["requests"] == 0
    finally:
        adapter.close()


def test_vision_payload_preserves_image_order_and_excludes_provenance(monkeypatch, tmp_path):
    import base64
    import hashlib

    from PIL import Image

    from s1mb.data import ImageAsset
    from s1mb.vision import prepare_image

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    images = {}
    for i, color in enumerate(["red", "blue"], 1):
        path = tmp_path / f"source-secret-{i}.png"
        Image.new("RGB", (320, 280), color).save(path)
        images[f"image_{i}"] = ImageAsset(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    q = Question(
        id="private-question",
        task="choice",
        instructions="Which image is red?",
        options=[
            Option(id="private-a", description="image_1"),
            Option(id="private-b", description="image_2"),
        ],
    )
    case = InferenceCase(
        case_id="private-case",
        state={"images": ["image_1", "image_2"]},
        images=images,
        questions=[q],
    )

    def decide(request):
        parts = json.loads(request.content)["input"]
        assert json.loads(parts[0]["text"]) == case.state
        assert [parts[1]["text"], parts[3]["text"]] == ["image_1", "image_2"]
        for part, asset in zip([parts[2], parts[4]], images.values(), strict=True):
            assert part["type"] == "image" and part["resolution"] == "high"
            assert base64.b64decode(part["data"]) == prepare_image(asset)[0]
            assert asset.sha256 not in request.content.decode()
        assert "source-secret" not in request.content.decode()
        assert "private-" not in request.content.decode()
        return httpx.Response(
            200,
            json={
                "answers": {
                    "question": {
                        "type": "choice",
                        "probabilities": {"option_0": 1.0, "option_1": 0.0},
                    }
                }
            },
        )

    adapter = GeminiDecisionsAdapter()
    mock(adapter, decide)
    try:
        result = adapter.predict(case)[0]
        assert result.probabilities == {"private-a": 1.0, "private-b": 0.0}
        assert len(adapter.image_receipts[case.case_id]) == 2
    finally:
        adapter.close()


def test_box_does_not_make_http_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    adapter = GeminiDecisionsAdapter()
    try:
        case = InferenceCase(
            case_id="c",
            state={},
            questions=[Question(id="q", task="box", instructions="Find the target.")],
        )
        assert adapter.predict(case)[0].error is not None
        assert adapter.usage["requests"] == 0
    finally:
        adapter.close()


def test_missing_key_is_rejected(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiDecisionsAdapter()


@pytest.mark.parametrize(
    "status,error,refused",
    [
        (400, {"code": "prohibited_content", "message": "upstream text"}, True),
        (400, {"code": "safety"}, True),
        (400, {"code": 400, "message": "upstream text", "status": "INVALID_ARGUMENT"}, False),
        (403, {"code": "prohibited_content"}, False),
    ],
)
def test_only_documented_block_reasons_are_refusals(monkeypatch, status, error, refused):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    adapter = GeminiDecisionsAdapter()
    mock(adapter, lambda request: httpx.Response(status, json={"error": error}))
    try:
        result = adapter.predict(
            InferenceCase(case_id="c", state="hello", questions=[noul_question()])
        )[0]
        assert result.probabilities is None
        if refused:
            assert result.abstention == "refusal" and result.error is None
        else:
            assert result.abstention is None and result.error == f"Gemini Decisions HTTP {status}"
        assert adapter.usage["retries"] == 0
    finally:
        adapter.close()
