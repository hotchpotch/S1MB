"""Exercise the multimodal upstream bridge without a GPU or downloaded weights."""

import hashlib
from types import SimpleNamespace

import pytest
from PIL import Image

from s1mb.adapters.imajev import ImajevAdapter
from s1mb.data import ImageAsset, InferenceCase, Option, Question


@pytest.mark.parametrize("image_count", [2, 3])
def test_images_reach_native_processor_in_declared_order(tmp_path, monkeypatch, image_count):
    images = {}
    for index, color in enumerate(["red", "green", "blue"][:image_count], start=1):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", (320, 320), color).save(path)
        images[f"image_{index}"] = ImageAsset(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    case = InferenceCase(
        case_id="private-id",
        state={"images": list(images)},
        images=images,
        questions=[Question(
            id="q", task="choice", instructions="Which image is red?",
            options=[Option(id="first", description="image_1"),
                     Option(id="second", description="image_2")],
        )],
    )
    captured = {}

    class ProcessorReached(Exception):
        pass

    def prepare(actual_images, prompt, labels):
        captured.update(images=actual_images, prompt=prompt, labels=labels)
        raise ProcessorReached

    def labels(count, native_image_count):
        assert count == 3 and native_image_count == image_count
        return ["A", "B", "C"]

    constructor = SimpleNamespace(model_construct=lambda **kwargs: SimpleNamespace(**kwargs))
    adapter = object.__new__(ImajevAdapter)
    adapter.image_receipts = {}
    adapter.max_images = image_count
    monkeypatch.setattr(
        adapter, "schema", SimpleNamespace(Option=constructor, ChoiceField=constructor), raising=False
    )
    adapter.native = SimpleNamespace(
        compile_question=lambda field, state, layout: (
            field.question + str(state), ["first", "second", "unknown"],
            ["image_1", "image_2", "Insufficient evidence"],
        )
    )
    adapter.engine = SimpleNamespace(prompt_layout="standard", labels=labels, prepare=prepare)
    with pytest.raises(ProcessorReached):
        adapter.predict(case)
    assert [image.getpixel((0, 0)) for image in captured["images"]] == [
        (255, 0, 0), (0, 128, 0), (0, 0, 255)
    ][:image_count]
    assert "private-id" not in captured["prompt"]
    assert [r["original_sha256"] for r in adapter.image_receipts[case.case_id]] == [
        asset.sha256 for asset in images.values()
    ]


def test_three_images_require_explicit_opt_in():
    adapter = object.__new__(ImajevAdapter)
    case = SimpleNamespace(questions=[], images=dict.fromkeys(["image_1", "image_2", "image_3"]))
    with pytest.raises(ValueError, match="at most 2 images"):
        adapter.predict(case)
