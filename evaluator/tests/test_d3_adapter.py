"""D3 native probability alignment and full structured criteria transport."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.d3 import D3Adapter


def test_native_typed_probabilities_and_original_numeric_anchors():
    adapter = object.__new__(D3Adapter)
    adapter.image_receipts = {}

    def decide(**request):
        assert request["state"] == cases().state
        assert request["images"] is None
        assert set(request["questions"]) == {"decision"}
        q = request["questions"]["decision"]
        if q["type"] == "noul":
            answer = {"type": "noul", "noul": 0.7}
        else:
            if q["type"] == "score":
                assert [v["value"] for v in q["criteria"]] == [10.0, 30.0]
            keys = list(q["criteria"]) if q["type"] == "choice" else ["0", "1"]
            answer = {"type": q["type"], "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}
        assert "gold-must-not-be-sent" not in str(request)
        return {"answers": {"decision": answer}}

    adapter.engine = SimpleNamespace(system_one=decide)
    predictions = adapter.predict(cases())
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities["true"] == 0.7
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}


def test_invalid_budget_before_loading():
    with pytest.raises(ValueError, match="positive"):
        D3Adapter("unused", "main", "unused", "cuda:0", context_limit=0)
