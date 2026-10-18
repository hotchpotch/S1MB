"""Explicit criteria survive the native Choice-only path and ordered image transport."""

import base64
import io

import pytest
from PIL import Image
from test_upstream_adapters import cases

from s1mb.adapters import e12ex_vision
from s1mb.adapters.e12ex_vision import E12exVisionAdapter


def test_authored_binary_criteria_and_numeric_anchors_reach_native_choice():
    adapter = E12exVisionAdapter.__new__(E12exVisionAdapter)
    adapter.image_receipts = {}
    seen = []

    def engine(state, questions):
        assert state == cases().state
        assert set(questions) == {"decision"}
        question = questions["decision"]
        assert question["type"] == "choice"
        assert "gold-must-not-be-sent" not in str((state, questions))
        criteria = question["criteria"]
        seen.append(criteria)
        return {"answers": {"decision": {"type": "choice", "probabilities":
            dict(zip(criteria, [0.2, 0.8], strict=True))}}}, None

    adapter.engine = engine
    result = adapter.predict(cases())
    assert seen[1] == {"true": "Yes", "false": "No"}
    assert [item["value"] for item in seen[2].values()] == [10.0, 30.0]
    assert result[0].probabilities == {"x": 0.2, "y": 0.8}
    assert result[1].probabilities["true"] == 0.2
    assert result[2].probabilities == {"low": 0.2, "high": 0.8}


def test_ordered_images_and_complete_context(monkeypatch):
    case = cases()
    images = [Image.new("RGB", (2, 3), color) for color in ["red", "blue"]]
    context = {"context": case.state, "visual_inputs": ["<image:1>", "<image:2>"]}
    monkeypatch.setattr(e12ex_vision, "ordered_images", lambda _: (images, context, [1, 2]))
    adapter = E12exVisionAdapter.__new__(E12exVisionAdapter)
    adapter.image_receipts = {}

    def engine(state, questions):
        assert state[0] == context
        assert len(state) == 3
        for part, image in zip(state[1:], images, strict=True):
            payload = base64.b64decode(part["image"].split(",", 1)[1])
            with Image.open(io.BytesIO(payload)) as decoded:
                assert decoded.tobytes() == image.tobytes()
        return {"answers": {"decision": {"type": "choice", "probabilities": {"true": 0.7, "false": 0.3}}}}, None

    adapter.engine = engine
    case.questions = [case.questions[1]]
    assert adapter.predict(case)[0].probabilities["true"] == 0.7


def test_budget_and_device_validation_before_loading():
    with pytest.raises(ValueError, match="positive"):
        E12exVisionAdapter("unused", "main", "unused", "cuda:0", "main", context_limit=0)
    with pytest.raises(ValueError, match="two CUDA"):
        E12exVisionAdapter("unused", "main", "unused", "cuda:0", "main", cuda_shards=1)
