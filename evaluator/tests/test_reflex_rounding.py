"""Reflex native decimal rounding must not hide invalid distributions."""

import pytest

from s1mb.adapters.reflex import normalize_native_probabilities


def test_many_options_restore_native_rounding_mass():
    raw = {str(i): round(1 / 77, 6) for i in range(77)}
    result = normalize_native_probabilities(raw, list(raw))
    assert sum(result.values()) == pytest.approx(1)
    assert list(result) == list(raw)
    assert all(value == pytest.approx(1 / 77) for value in result.values())


@pytest.mark.parametrize(
    "raw,keys",
    [
        ({"a": 0.4, "b": 0.4}, ["a", "b"]),
        ({"a": -0.1, "b": 1.1}, ["a", "b"]),
        ({"a": float("nan"), "b": 0.5}, ["a", "b"]),
        ({"a": 1.0}, ["a", "b"]),
        ({"a": 0.0, "b": 0.0}, ["a", "b"]),
    ],
)
def test_invalid_distributions_are_rejected(raw, keys):
    with pytest.raises(ValueError):
        normalize_native_probabilities(raw, keys)
