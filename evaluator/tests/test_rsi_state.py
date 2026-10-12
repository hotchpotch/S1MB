"""Structured evidence must not be rejected merely because it resembles a chat transcript."""

import json
from types import SimpleNamespace

import pytest

from s1mb.adapters.rsi import rsi_state_text


def test_unsupported_role_preserves_authored_json_and_other_errors_propagate():
    def unsupported(state):
        raise ValueError("Chat state has an unsupported message role")

    wire = SimpleNamespace(state_to_text=unsupported, RequestError=ValueError,
                           serialize=lambda state: json.dumps(state, separators=(",", ":")))
    evidence = [{"role": "arbitrator", "content": "Approved", "weight": 0.75}]
    assert json.loads(rsi_state_text(evidence, wire)) == evidence
    wire.state_to_text = lambda state: "native rendering"
    assert rsi_state_text(evidence, wire) == "native rendering"

    def invalid(state):
        raise ValueError("Jev state supports text content only")

    wire.state_to_text = invalid
    with pytest.raises(ValueError, match="text content only"):
        rsi_state_text(evidence, wire)
