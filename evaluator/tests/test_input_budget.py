"""State budgeting preserves decision fields and honors native request limits."""

import copy
import json

import pytest

from s1mb.adapters.input_budget import budget_state


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, ids, **kwargs):
        return "".join(ids)


def budget(state, limit, fraction=1.0):
    return budget_state(
        state,
        CharacterTokenizer(),
        json.dumps,
        lambda value: 20 + len(json.dumps(value)),
        limit,
        fraction,
    )


def test_fitting_state_is_unchanged():
    state = {"body": "small", "question": "preserve"}
    assert budget(state, 100).state is state


def test_shortening_preserves_structure_and_authored_fields():
    state = {
        "body": "HEAD" + "x" * 1000 + "TAIL",
        "question": "q" * 300,
        "criteria": ["c" * 300],
        "numbers": [1, 2],
    }
    original = copy.deepcopy(state)
    result = budget(state, 900)
    assert result.request_after <= 900
    assert result.state["body"].startswith("HEAD")
    assert result.state["body"].endswith("TAIL")
    for field in ("question", "criteria", "numbers"):
        assert result.state[field] == original[field]
    assert state == original


def test_budget_accumulates_shortening_across_body_fields():
    result = budget({"a": "x" * 600, "b": "y" * 600}, 150)
    assert result.request_after <= 150
    assert set(result.state) == {"a", "b"}


def test_explicit_fraction_is_a_hard_bound():
    with pytest.raises(ValueError, match="allowed"):
        budget("x" * 1000, 500, 0.05)
    result = budget("x" * 1000, 1000, 0.05)
    assert result.state_before - result.state_after <= int(result.state_before * 0.05)


def test_protected_content_cannot_be_removed_to_force_fit():
    with pytest.raises(ValueError):
        budget({"question": "q" * 1000, "body": "x" * 300}, 400)


@pytest.mark.parametrize("fraction", [0, -1, 1.01, float("nan")])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        budget("x" * 1000, 500, fraction)
