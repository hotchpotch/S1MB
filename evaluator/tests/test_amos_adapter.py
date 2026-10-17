"""Native typed criteria, probability alignment and ordered image transport."""

import base64
import io

import pytest
from PIL import Image
from test_upstream_adapters import cases

from s1mb.adapters import amos
from s1mb.adapters.amos import AmosAdapter


def test_native_criteria_alignment_and_no_target_leakage():
    adapter = AmosAdapter.__new__(AmosAdapter)
    adapter.image_receipts = {}

    def engine(request):
        assert request["state"] == cases().state
        assert request["images"] == []
        assert set(request["questions"]) == {"decision"}
        assert "gold-must-not-be-sent" not in str(request)
        question = request["questions"]["decision"]
        if question["type"] == "noul":
            assert question["criteria"] == {"true": "Yes", "false": "No"}
            answer = {"type": "noul", "noul": 0.7}
        else:
            if question["type"] == "score":
                assert [item["value"] for item in question["criteria"]] == [10.0, 30.0]
                keys = ["0", "1"]
            else:
                keys = list(question["criteria"])
            answer = {"type": question["type"], "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}
        return {"answers": {"decision": answer}}

    adapter.engine = engine
    result = adapter.predict(cases())
    assert result[0].probabilities == {"x": 0.2, "y": 0.8}
    assert result[1].probabilities["true"] == 0.7
    assert result[2].probabilities == {"low": 0.2, "high": 0.8}


def test_ordered_pixels_without_source_paths(monkeypatch):
    images = [Image.new("RGB", (2, 3), color) for color in ["red", "blue"]]
    case = cases()
    context = {"context": case.state, "visual_inputs": ["<image:1>", "<image:2>"]}
    monkeypatch.setattr(amos, "ordered_images", lambda _: (images, context, [1, 2]))
    adapter = AmosAdapter.__new__(AmosAdapter)
    adapter.image_receipts = {}

    def engine(request):
        assert request["state"] == context
        assert len(request["images"]) == 2
        for payload, image in zip(request["images"], images, strict=True):
            with Image.open(io.BytesIO(base64.b64decode(payload))) as decoded:
                assert decoded.tobytes() == image.tobytes()
        return {"answers": {"decision": {"type": "noul", "noul": 0.7}}}

    adapter.engine = engine
    case.questions = [case.questions[1]]
    assert adapter.predict(case)[0].probabilities["true"] == 0.7


def test_invalid_budget_and_placement_before_loading():
    with pytest.raises(ValueError, match="positive"):
        AmosAdapter("unused", "main", "unused", "cuda:0", context_limit=0)
    with pytest.raises(ValueError, match="cuda_shards"):
        AmosAdapter("unused", "main", "unused", "cuda:0", cuda_shards=0)
