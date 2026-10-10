"""Native Jevstral overflow must be rejected before inference."""
from types import SimpleNamespace

import pytest

from s1mb.adapters.jevstral import JevstralAdapter
from s1mb.data import InferenceCase


def test_overflow_admission_precedes_inference():
    def rows(record, limits):
        raise ValueError("native row exceeds serving limit")

    adapter = JevstralAdapter.__new__(JevstralAdapter)
    adapter.native = SimpleNamespace(to_request=lambda *args: None)
    adapter.engine = SimpleNamespace(encoder=SimpleNamespace(rows=rows))
    adapter.limits = None
    case = InferenceCase.model_validate({
        "case_id": "example", "state": "state",
        "questions": [{"id": "pick", "task": "choice", "instructions": "Choose",
                       "options": [{"id": "a", "description": "First"},
                                   {"id": "b", "description": "Second"}]}],
    })
    with pytest.raises(ValueError, match="native row"):
        adapter.predict(case)
