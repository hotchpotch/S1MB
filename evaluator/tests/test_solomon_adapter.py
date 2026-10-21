"""Solomon's gateway must preserve authored criteria and duplicate candidate positions."""

import json

from s1mb.adapters.sifr import sifr_questions
from s1mb.adapters.solomon import solomon_options


def test_structured_score_values_and_duplicate_positions():
    from test_upstream_adapters import cases

    case = cases()
    case.questions[2].options.reverse()
    wire = sifr_questions(case)
    values = [json.loads(value) for value in solomon_options(wire["rate"])]
    assert [value["value"] for value in values] == [30, 10]
    duplicate = {"criteria": {"option_0": "same", "option_1": "same", "option_2": "different"}}
    assert solomon_options(duplicate) == [
        "option_0: same", "option_1: same", "option_2: different",
    ]
    assert solomon_options({"criteria": {"option_0": "first", "option_1": "second"}}) == [
        "first", "second",
    ]


def test_images_keep_order_and_are_available_until_native_scoring_finishes(tmp_path, monkeypatch):
    import tempfile
    from pathlib import Path

    from PIL import Image
    from test_upstream_adapters import cases

    import s1mb.adapters.solomon as module

    case = cases()
    images = [Image.new("RGB", (10, 10), "red"), Image.new("RGB", (10, 10), "blue")]
    state = {"context": case.state, "visual_inputs": {"left": "<image:1>", "right": "<image:2>"}}
    monkeypatch.setattr(module, "ordered_images", lambda _: (images, state, [{"index": 1}, {"index": 2}]))
    adapter = object.__new__(module.SolomonAdapter)
    adapter.store = tempfile.TemporaryDirectory(dir=tmp_path)
    adapter.image_receipts = {}
    seen_paths = []

    def score(received, parts, actual_images):
        assert received is case and actual_images is images
        assert json.loads(parts[0]["text"]) == state
        assert [p["text"] for p in parts[1:] if "text" in p] == ["<image:1>", "<image:2>"]
        seen_paths.extend(Path(p["image"]) for p in parts if "image" in p)
        assert [Image.open(p).getpixel((0, 0)) for p in seen_paths] == [(255, 0, 0), (0, 0, 255)]
        return ["native-result"]

    monkeypatch.setattr(adapter, "_predict_parts", score)
    try:
        assert adapter.predict(case) == ["native-result"]
        assert all(not p.exists() for p in seen_paths)
        assert adapter.image_receipts[case.case_id] == [{"index": 1}, {"index": 2}]
    finally:
        adapter.store.cleanup()


def test_expanded_image_context_rejected_before_prefill():
    from types import SimpleNamespace

    import pytest
    from test_upstream_adapters import cases

    from s1mb.adapters.solomon import SolomonAdapter

    adapter = object.__new__(SolomonAdapter)
    adapter.native = SimpleNamespace(listwise_block=lambda *a, **k: ("native block", 2))
    adapter.engine = SimpleNamespace(
        _render=lambda *a: "native rendered prompt",
        processor=lambda **k: {"input_ids": SimpleNamespace(shape=(1, 32769))},
        prefill=lambda *a: pytest.fail("Overflow must be rejected before prefill"),
    )
    with pytest.raises(ValueError, match="exceeds 32768"):
        adapter._predict_parts(cases(), [{"text": "full state"}], [object()])
