"""OneJev transport preserves authored values and native answer alignment."""

from types import ModuleType, SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.onejev import OneJevAdapter, onejev_questions


@pytest.mark.parametrize("shards", [0, 3])
def test_invalid_cuda_shards_before_loading(shards):
    with pytest.raises(ValueError, match="cuda_shards"):
        OneJevAdapter("unused", "main", "unused", "cuda:0", cuda_shards=shards)


def test_structured_criteria_numeric_score_and_no_identifiers():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    case.questions[2].options.reverse()
    questions = onejev_questions(case)
    assert questions["choose"]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["criteria"] == [
        {"value": 30.0, "description": "High"},
        {"value": 10.0, "description": "Low"},
    ]
    assert "gold-must-not-be-sent" not in str(questions)


def test_single_native_request_alignment_and_no_media():
    adapter = OneJevAdapter.__new__(OneJevAdapter)
    adapter.schema = ModuleType("synthetic_qev_schema")
    adapter.schema.__dict__["SystemOneRequest"] = SimpleNamespace(
        model_validate=lambda request: request
    )

    def decide(request, *, debias, media):
        assert debias == 1 and media == []
        assert request["state"] == cases().state
        assert set(request["questions"]) == {"decision"}
        question = request["questions"]["decision"]
        task = question["type"]
        if task == "noul":
            answer = {"type": task, "noul": 0.7}
        else:
            keys = list(question["criteria"]) if task == "choice" else ["0", "1"]
            answer = {"type": task, "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}
        return SimpleNamespace(model_dump=lambda **kwargs: {"answers": {"decision": answer}}), {}

    adapter.engine = SimpleNamespace(decide=decide)
    predictions = adapter.predict(cases())
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities["true"] == 0.7
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}


def test_invalid_budget_before_loading():
    with pytest.raises(ValueError, match="positive"):
        OneJevAdapter("unused", "main", "unused", "cuda:0", context_limit=0)


@pytest.mark.parametrize("count", [2, 3])
def test_ordered_images_and_neutral_references(tmp_path, count):
    import hashlib

    from PIL import Image

    from s1mb.data import ImageAsset, InferenceCase

    images = {}
    colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255)][:count]
    for index, color in enumerate(colors):
        path = tmp_path / f"private-source-{index}.png"
        Image.new("RGB", (320, 320), color).save(path)
        images[f"image_{index + 1}"] = ImageAsset(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    case = InferenceCase(
        case_id="private-id", state={"context": "Inspect all images."},
        questions=cases().questions[:1], images=images,
    )
    adapter = object.__new__(OneJevAdapter)
    adapter.image_receipts = {}
    adapter.schema = ModuleType("synthetic_image_schema")
    adapter.schema.__dict__["SystemOneRequest"] = SimpleNamespace(model_validate=lambda r: r)

    class Reached(Exception):
        pass

    def decide(request, *, debias, media):
        assert [m["image"].getpixel((0, 0)) for m in media] == colors
        assert request["state"]["context"] == case.state
        assert request["state"]["visual_inputs"] == {
            name: f"<image:{i}>" for i, name in enumerate(images, 1)
        }
        assert "private" not in str(request)
        raise Reached

    adapter.engine = SimpleNamespace(decide=decide)
    with pytest.raises(Reached):
        adapter.predict(case)
    assert [r["original_sha256"] for r in adapter.image_receipts[case.case_id]] == [
        a.sha256 for a in images.values()
    ]
