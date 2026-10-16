"""Synthetic image fixtures exercise payloads and coordinate transformations offline."""

import hashlib
import json
from email import policy
from email.parser import BytesParser

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from s1mb.adapters.typesafe import TypeSafeAdapter
from s1mb.data import Box, ImageAsset, InferenceCase, Prediction, Question, check_prediction
from s1mb.metrics import box_iou
from s1mb.vision import original_box, prepare_image


def asset(tmp_path, size=(100, 1600), image_policy="scene-896"):
    path = tmp_path / "synthetic.png"
    Image.new("RGB", size, "red").save(path)
    return ImageAsset(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), policy=image_policy
    )


def test_padding_box_round_trip_and_checksum(tmp_path):
    image = asset(tmp_path)
    payload, mime, transform = prepare_image(image)
    assert mime == "image/png"
    assert transform["payload_size"] == [280, 896]
    assert transform["content_rectangle"] == [112, 0, 56, 896]
    content = Box(left=112 / 280, top=0, right=168 / 280, bottom=1)
    assert original_box(content, transform) == Box(left=0, top=0, right=1, bottom=1)
    assert prepare_image(image)[0] == payload
    image.sha256 = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        prepare_image(image)


def test_original_preserves_iqa_bytes_and_rejects_invalid_dimensions(tmp_path):
    image = asset(tmp_path, (1024, 768), "original")
    assert prepare_image(image)[0] == (tmp_path / "synthetic.png").read_bytes()
    with pytest.raises(ValueError, match="limits"):
        prepare_image(asset(tmp_path, (100, 200), "original"))


def test_iou_and_invalid_box_prediction():
    full = Box(left=0, top=0, right=1, bottom=1)
    half = Box(left=0, top=0, right=0.5, bottom=1)
    assert box_iou(full, half) == 0.5
    with pytest.raises(ValidationError):
        Box(left=1, top=0, right=0, bottom=1)
    with pytest.raises(ValueError, match="Distribution"):
        check_prediction(
            Prediction(case_id="c", question_id="q", probabilities={}),
            Question(id="q", task="box", instructions="Find it"),
        )


def test_multipart_neutral_references_and_box_mapping(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    adapter = TypeSafeAdapter("jev-1.14.0")
    adapter.client.close()
    image = asset(tmp_path)
    case = InferenceCase(
        case_id="private-source-id",
        state={"images": ["image_1"], "expression": "the red rectangle"},
        images={"image_1": image},
        questions=[Question(id="q", task="box", instructions="Locate the rectangle in image_1.")],
    )

    def handler(request):
        assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: "
            + request.headers["content-type"].encode()
            + b"\r\n\r\n"
            + request.content
        )
        parts = list(message.iter_parts())
        decoded = parts[0].get_payload(decode=True)
        assert isinstance(decoded, bytes)
        body = json.loads(decoded)
        assert len(parts) == 2
        assert body["questions"]["q"] == {
            "type": "box",
            "instructions": "Locate the rectangle in image_1.",
        }
        ref = body["state"]["images"]["image_1"]
        assert body["assets"][ref] == {"type": "image", "part": "image_1"}
        assert (
            json.dumps(body).count(ref) == 2
        )  # Asset definition and exactly one model-facing use.
        assert "private-source-id" not in json.dumps(body)
        assert image.sha256 not in json.dumps(body)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "answers": {
                    "q": {
                        "type": "box",
                        "box": {"left": 0.4, "right": 0.6, "top": 0, "bottom": 1},
                        "confidence": 0.8,
                    }
                },
            },
        )

    adapter.client = httpx.Client(base_url=adapter.base_url, transport=httpx.MockTransport(handler))
    try:
        prediction = adapter.predict(case)[0]
        assert prediction.box is not None
        assert prediction.box.left == pytest.approx(0)
        assert prediction.box.right == pytest.approx(1)
        assert adapter.image_receipts[case.case_id][0]["payload_size"] == [280, 896]
    finally:
        adapter.close()


def test_wide_score_is_rejected_before_http(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    adapter = TypeSafeAdapter("jev-1.14.0")
    case = InferenceCase(
        case_id="count",
        state={},
        questions=[
            Question(
                id="q",
                task="score",
                instructions="Estimate the count.",
                options=[{"id": str(i), "value": i, "description": str(i)} for i in range(11)],
            )
        ],
    )
    try:
        with pytest.raises(ValueError, match="at most 10"):
            adapter.predict(case)
        assert adapter.usage["requests"] == 0
    finally:
        adapter.close()


def test_box_run_round_trip_rejects_tampered_metrics(tmp_path):
    from s1mb.data import Benchmark, Case, ModelInfo, validate_result, write_json
    from s1mb.runner import evaluate

    directory = tmp_path / "datasets" / "synthetic"
    directory.mkdir(parents=True)
    image = asset(directory, (280, 280))
    full = Box(left=0, top=0, right=1, bottom=1)
    cases = [
        Case(
            case_id=f"c{i}",
            group_id="g",
            language="en",
            state={"images": ["image_1"]},
            images={"image_1": image.model_copy(update={"path": "synthetic.png"})},
            questions=[Question(id="q", task="box", instructions="Find the red rectangle.")],
            targets={"q": {"kind": "synthetic", "box": full}},
            provenance={"input_hash": str(i)},
        )
        for i in range(2)
    ]
    write_json(directory / "vision-cases.json", [c.model_dump() for c in cases])
    benchmark = Benchmark(
        id="synthetic-box",
        task="box",
        dataset="datasets/synthetic",
        split="test",
        case_count=2,
        decision_count=2,
        primary_metric="mean_iou",
    )
    write_json(tmp_path / "benchmarks/synthetic-box.json", benchmark.model_dump())

    class Adapter:
        supports_images = True

        def predict(self, case):
            assert not hasattr(case, "targets")
            assert not hasattr(case, "provenance")
            return [
                Prediction(
                    case_id=case.case_id,
                    question_id="q",
                    box=Box(left=0, top=0, right=0.5, bottom=1),
                )
            ]

        def metadata(self):
            return ModelInfo(id="synthetic", adapter="test", revision="fixture")

        def close(self):
            pass

    result = evaluate(Adapter(), tmp_path, benchmark, tmp_path / "results", "box-smoke", limit=1)
    assert result.counts.succeeded == 1
    assert result.status == "partial"
    assert result.metrics == {"mean_iou": 0.5, "accuracy_at_iou_0_5": 1.0}
    validate_result(tmp_path, result)
    result.metrics["mean_iou"] = 1.0
    with pytest.raises(ValueError, match="recomputed"):
        validate_result(tmp_path, result)


def test_degenerate_prediction_is_a_miss_but_target_requires_positive_area():
    from s1mb.data import Target

    point = Box(left=0.03, top=0.63, right=0.03, bottom=0.67)
    target = Box(left=0, top=0, right=1, bottom=1)
    assert box_iou(point, target) == 0
    assert box_iou(point, point) == 0
    with pytest.raises(ValidationError, match="Target box must have positive area"):
        Target(kind="box", box=point)
