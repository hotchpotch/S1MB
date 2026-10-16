"""Vev native transport preserves typed questions and ordered image payloads."""

import base64
import hashlib
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from test_upstream_adapters import cases

from s1mb.adapters.vev import VevAdapter
from s1mb.data import ImageAsset


def test_native_question_values_and_image_order(tmp_path):
    case = cases()
    case.images = {}
    for i, color in enumerate([(255, 0, 0), (0, 0, 255)]):
        path = tmp_path / f"secret-{i}.png"
        Image.new("RGB", (320, 320), color).save(path)
        case.images[f"image_{i + 1}"] = ImageAsset(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    adapter = object.__new__(VevAdapter)
    adapter.context_limit = 32768
    adapter.image_receipts = {}

    def run(state, questions, **kwargs):
        assert state["context"] == case.state
        pixels = []
        for ref in state["visual_inputs"].values():
            with Image.open(io.BytesIO(base64.b64decode(ref["image"]["url"].split(",", 1)[1]))) as image:
                pixels.append(image.getpixel((0, 0)))
        assert pixels == [(255, 0, 0), (0, 0, 255)]
        assert "secret" not in str(state)
        assert kwargs["max_request_tokens"] == 32768 and not kwargs["allow_remote"]
        q = questions["decision"]
        if q["type"] == "noul":
            answer = {"type": "noul", "noul": 0.6}
        else:
            if q["type"] == "score":
                assert [o["value"] for o in q["criteria"]] == [10.0, 30.0]
            keys = list(q["criteria"]) if q["type"] == "choice" else ["0", "1"]
            answer = {"type": q["type"], "probabilities": dict(zip(keys, [0.3, 0.7], strict=True))}
        return SimpleNamespace(answers={"decision": answer})

    adapter.engine = SimpleNamespace(run=run)
    predictions = adapter.predict(case)
    assert predictions[0].probabilities == {"x": 0.3, "y": 0.7}
    assert predictions[1].probabilities["true"] == 0.6
    assert predictions[2].probabilities == {"low": 0.3, "high": 0.7}
    assert len(adapter.image_receipts[case.case_id]) == 2


def test_invalid_budget_before_loading():
    with pytest.raises(ValueError, match="positive"):
        VevAdapter("unused", "main", "unused", "cuda:0", context_limit=0)
