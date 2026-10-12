"""Complete native rows must be admitted before GPU inference."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.sieve_9b import Sieve9BAdapter
from s1mb.data import InferenceCase


def test_context_overflow_precedes_native_forward():
    adapter = Sieve9BAdapter.__new__(Sieve9BAdapter)
    adapter.native = SimpleNamespace(to_record=lambda *args: ("state", [], None))
    adapter.engine = SimpleNamespace(encode=lambda *args, **kwargs: {"ids": [1] * 65})
    adapter.limit = 64
    case = InferenceCase.model_validate({
        "case_id": "example", "state": "state",
        "questions": [{"id": "pick", "task": "choice", "instructions": "Choose",
                       "options": [{"id": "a", "description": "First"},
                                   {"id": "b", "description": "Second"}]}],
    })
    with pytest.raises(ValueError, match="Complete Sieve-9B row"):
        adapter.predict(case)
