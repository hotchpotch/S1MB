"""Admission occurs before native model inference."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.ajev import AJevAdapter
from s1mb.data import InferenceCase


def test_complete_prompt_overflow_precedes_forward(monkeypatch):
    adapter = AJevAdapter.__new__(AJevAdapter)
    adapter.native = None
    monkeypatch.setattr(adapter, "request", None, raising=False)
    monkeypatch.setattr(adapter, "predictor", SimpleNamespace(
        prompt_ids=lambda *args: list(range(65)),
    ), raising=False)
    adapter.engine = SimpleNamespace(tok=None)
    adapter.limit = 64
    monkeypatch.setattr("s1mb.adapters.ajev.ajev_decisions", lambda *args: [object()])
    case = InferenceCase.model_validate({
        "case_id": "example", "state": "state", "questions": [{
            "id": "pick", "task": "choice", "instructions": "Choose",
            "options": [{"id": "a", "description": "First"},
                        {"id": "b", "description": "Second"}],
        }],
    })
    with pytest.raises(ValueError, match="Complete AJev prompt"):
        adapter.predict(case)
