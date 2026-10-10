"""Native question rendering and exact alignment of typed probabilities."""

import json
from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.jev_style import JevStyleAdapter, jev_style_questions


def test_native_fields_preserve_authored_structured_and_numeric_values():
    case = cases()
    case.questions[0].instructions_json = '{"instruction":"Pick a team"}'
    case.questions[0].options[0].description_json = '{"nested":true}'
    case.questions[2].options.reverse()
    questions = jev_style_questions(case)
    assert json.loads(questions["choose"]["ins"]) == {
        "system": "System.", "instruction": {"instruction": "Pick a team"}
    }
    assert questions["choose"]["crit"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["crit"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["crit"] == [
        {"value": 30.0, "description": "High"}, {"value": 10.0, "description": "Low"}
    ]
    assert "gold-must-not-be-sent" not in str(questions)


def test_native_probability_keys_align_with_declared_option_order():
    adapter = JevStyleAdapter.__new__(JevStyleAdapter)

    def decide(state, question):
        assert state == cases().state
        assert set(question) == {"t", "ins", "crit"}
        keys = list(question["crit"]) if question["t"] != "score" else ["0", "1"]
        return {"probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}

    adapter.engine = SimpleNamespace(decide=decide)
    predictions = adapter.predict(cases())
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities == {"true": 0.2, "false": 0.8}
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}


@pytest.mark.parametrize("probabilities", [{"other": 1}, {"option_0": 0.2, "option_1": 0.3}])
def test_native_invalid_alignment_or_distribution_rejected(probabilities):
    adapter = JevStyleAdapter.__new__(JevStyleAdapter)
    adapter.engine = SimpleNamespace(decide=lambda *args: {"probabilities": probabilities})
    with pytest.raises(ValueError):
        adapter.predict(cases())


def test_invalid_budget_rejected_before_checkpoint_load():
    with pytest.raises(ValueError, match="positive"):
        JevStyleAdapter("unused", "main", "unused", "cuda:0", context_limit=0)
