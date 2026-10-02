"""Structured records with roles are not necessarily chat messages."""

import json
from types import SimpleNamespace

import pytest

from s1mb.adapters.jevone import render_native_state


@pytest.mark.parametrize("wrapped", [False, True])
def test_role_records_without_content_preserve_entire_state(wrapped):
    records = [{"role": "reviewer", "rating": 3, "evidence": {"amount": 42}}]
    state = {"messages": records, "context": "retain"} if wrapped else records

    def reject_native(value):
        raise AssertionError("Malformed chat must not enter the native chat renderer")

    rendered = render_native_state(SimpleNamespace(render_state=reject_native), state)
    assert json.loads(rendered) == state


def test_valid_chat_keeps_native_rendering():
    state = [{"role": "user", "content": "Evidence"}]
    calls = []

    def native(value):
        calls.append(value)
        return "USER: Evidence"

    assert render_native_state(SimpleNamespace(render_state=native), state) == "USER: Evidence"
    assert calls == [state]


def test_non_string_role_preserves_structured_content():
    state = [{"role": 7, "content": {"facts": [1, 2]}}]
    rendered = render_native_state(SimpleNamespace(), state)
    assert json.loads(rendered) == state
