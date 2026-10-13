"""Saihai must match its released native mapping without exposing benchmark identifiers."""

import json
from types import SimpleNamespace

import pytest

from s1mb.adapters.saihai import SaihaiAdapter, saihai_input


def test_exact_mapping_preserves_structured_values_and_numeric_order():
    from test_upstream_adapters import cases

    case = cases()
    choice, noul, score = case.questions
    choice.instructions_json = json.dumps({"task": "Choose"})
    choice.options[0].description_json = json.dumps({"meaning": ["first", 2]})
    score.options.reverse()
    item, options = saihai_input(case, choice)
    assert item["state"] == case.state
    assert item["instructions"] == {"system": "System.", "instruction": {"task": "Choose"}}
    assert item["options"] == [
        {"label": "option_0", "description": {"meaning": ["first", 2]}},
        {"label": "option_1", "description": "Same"},
    ]
    assert [option.id for option in options] == ["x", "y"]
    item, options = saihai_input(case, noul)
    assert [option.id for option in options] == ["false", "true"]
    assert item["options"] == [
        {"label": "false", "description": "No"}, {"label": "true", "description": "Yes"},
    ]
    item, options = saihai_input(case, score)
    assert [option["value"] for option in item["options"]] == [30, 10]
    assert [option.id for option in options] == ["high", "low"]
    assert "boundary" not in json.dumps(item)
    assert "targets" not in item and "provenance" not in item


def test_prediction_alignment_and_native_errors():
    from test_upstream_adapters import cases

    adapter = SaihaiAdapter.__new__(SaihaiAdapter)
    adapter.engine = SimpleNamespace(predict=lambda item: {"probabilities": [0.2, 0.8]})
    predictions = adapter.predict(cases())
    assert predictions[1].probabilities == {"false": 0.2, "true": 0.8}
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}
    adapter.engine.predict = lambda item: {"probabilities": [0.2, 0.2]}
    with pytest.raises(ValueError):
        adapter.predict(cases())
    adapter.engine.predict = lambda item: {"probabilities": [1.0]}
    with pytest.raises(ValueError):
        adapter.predict(cases())
    def overflow(item):
        raise ValueError("Input exceeds contract context limit; refusing truncation")
    adapter.engine.predict = overflow
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())
