"""Image Hopper preserves authored criteria, numeric anchors and image-only contract."""

import json
from types import SimpleNamespace

import pytest
from PIL import Image
from test_upstream_adapters import cases

from s1mb.adapters.image_hopper import ImageHopperAdapter


def test_authored_choice_transport_and_score_index_alignment(monkeypatch):
    case = cases()
    image = Image.new("RGB", (2, 2))
    monkeypatch.setattr("s1mb.adapters.image_hopper.ordered_images",
                        lambda c: ([image], {"context": c.state}, [{"verified": True}]))
    requests = []

    def native(body):
        requests.append(body)
        assert body["images"][0].startswith("data:image/png;base64,")
        q = body["questions"]["decision"]
        assert q["type"] == "choice"
        if len(requests) == 2:
            assert list(q["criteria"].values()) == ["No", "Yes"]
        if len(requests) == 3:
            assert [json.loads(v)["value"] for v in q["criteria"].values()] == [10, 30]
        return {"answers": {"decision": {"type": "choice", "probabilities": {
            "option_0": 0.2, "option_1": 0.8,
        }}}}

    adapter = object.__new__(ImageHopperAdapter)
    adapter.image_receipts = {}
    adapter.engine = native
    predictions = adapter.predict(case)
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities["true"] == 0.8
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}
    assert "gold-must-not-be-sent" not in str(requests)


def test_text_scope_rejected_without_synthetic_image():
    adapter = object.__new__(ImageHopperAdapter)
    adapter.engine = SimpleNamespace()
    with pytest.raises(ValueError, match="requires at least one image"):
        adapter.predict(cases())


def test_native_receipts_keep_digests_without_local_paths():
    from s1mb.adapters.image_hopper import public_hopper_provenance

    native = {"adapter": {"path": "/private/adapter", "sha256": "a" * 64},
              "base_revision": "b" * 40, "routes": [{"path": "/private/route", "cap": 768}]}
    public = public_hopper_provenance(native)
    assert public == {"adapter": {"sha256": "a" * 64}, "base_revision": "b" * 40,
                      "routes": [{"cap": 768}]}
    assert native["adapter"]["path"] == "/private/adapter"
