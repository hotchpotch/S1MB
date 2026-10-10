"""Julia boundary tests without weights or inference hardware."""

import json
from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.julia import JuliaAdapter, julia_row


def adapter(values=None, reject=False):
    model = JuliaAdapter.__new__(JuliaAdapter)
    model.context_limit = 8192

    def sequence(tokenizer, row, limit, head, *, strict):
        assert (limit, head, strict) == (8192, 512, True)
        assert set(row) == {"state", "question", "type", "options"}
        if reject:
            raise ValueError("Game state exceeds lossless context budget")
        return {"ids": [1, 2]}

    def logits(rows):
        assert rows[0]["_encoded"] == {"ids": [1, 2]}
        return [[0.0, 0.0]] if values is None else values

    model.native = SimpleNamespace(validate_row=lambda *args: None, sequence=sequence)
    model.engine = SimpleNamespace(tokenizer=None, logits=logits)
    return model


def test_rendering_preserves_structured_values_and_numeric_rubric():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    row, keys = julia_row(case, case.questions[0])
    assert json.loads(row["options"][0]) == {"nested": True}
    assert keys == ["x", "y"]
    assert json.loads(row["question"]) == {"system": "System.", "instruction": "Choose."}
    assert "boundary" not in str(row)
    assert "gold-must-not-be-sent" not in str(row)
    case.questions[2].options.reverse()
    row, keys = julia_row(case, case.questions[2])
    assert keys == ["high", "low"]
    assert [json.loads(v)["value"] for v in row["options"]] == [30.0, 10.0]


def test_authored_noul_order_and_probability_alignment():
    case = cases()
    row, keys = julia_row(case, case.questions[1])
    assert keys == ["false", "true"]
    assert row["options"] == ["No", "Yes"]
    predictions = adapter([[0.0, 1000.0]]).predict(case)
    assert predictions[1].probabilities == {"false": 0.0, "true": 1.0}
    assert predictions[2].probabilities == {"low": 0.0, "high": 1.0}


def test_strict_overflow_rejected_before_inference():
    model = adapter(reject=True)
    model.engine.logits = lambda rows: pytest.fail("Inference must not run")
    with pytest.raises(ValueError, match="lossless context"):
        model.predict(cases())


@pytest.mark.parametrize("values", [[], [[0.0]], [[0.0, float("nan")]], [[float("inf"), 0]]])
def test_invalid_native_logits_rejected(values):
    with pytest.raises(ValueError):
        adapter(values).predict(cases())


@pytest.mark.parametrize("device,limit", [("cpu", None), ("cuda:0", 0), ("cuda:0", 8193)])
def test_bad_device_and_context_rejected_without_download(device, limit):
    with pytest.raises(ValueError):
        JuliaAdapter("unused", "main", device, context_limit=limit)
