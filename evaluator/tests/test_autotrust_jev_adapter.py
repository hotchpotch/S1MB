"""Slot-head admission and typed fidelity without private data or GPU."""

import pytest

from s1mb.adapters.autotrust_jev import autotrust_choice_text


def test_native_template_preserves_numeric_levels_and_structured_definitions():
    text = autotrust_choice_text({"content": "state"}, {"question": "Choose"}, {
        "option_0": {"value": -2, "description": {"meaning": "Low"}},
        "option_1": {"value": 9, "description": "High"},
    })
    assert text == (
        '[kind] choice\n[state] {"content": "state"}\n[question] {"question": "Choose"}'
        '\n[options]\nA) {"value": -2, "description": {"meaning": "Low"}}'
        '\nB) {"value": 9, "description": "High"}\n[decision]:'
    )


def test_native_slot_capacity_does_not_discard_candidates():
    with pytest.raises(ValueError, match="2–16"):
        autotrust_choice_text("state", "Choose", {str(i): "candidate" for i in range(17)})
