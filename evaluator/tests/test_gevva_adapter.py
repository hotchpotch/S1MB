"""Complete-input admission must precede native Gevva inference."""
from types import SimpleNamespace

import pytest

from s1mb.adapters.gevva import GevvaAdapter
from s1mb.data import InferenceCase


def test_native_pair_truncation_is_rejected_before_inference():
    adapter = GevvaAdapter.__new__(GevvaAdapter)
    adapter.native = SimpleNamespace(
        tokenize_nli_pair_safe=lambda tok, p, h, cap: [1] if cap == 64 else [1, 2],
    )
    adapter.engine = SimpleNamespace(tokenizer=None, max_length=64)
    case = InferenceCase.model_validate({
        "case_id": "example", "state": "state",
        "questions": [{"id": "pick", "task": "choice", "instructions": "Choose",
                       "options": [{"id": "a", "description": "First"},
                                   {"id": "b", "description": "Second"}]}],
    })
    with pytest.raises(ValueError, match="pair budget"):
        adapter.predict(case)


def test_duplicate_descriptions_keep_distinct_native_hypotheses():
    seen = []

    def decide(context, instruction, candidates):
        assert candidates == ["option_0: Same", "option_1: Same"]
        return SimpleNamespace(scores=[0.4, 0.6])

    adapter = GevvaAdapter.__new__(GevvaAdapter)
    adapter.native = SimpleNamespace(
        tokenize_nli_pair_safe=lambda tok, p, h, cap: seen.append(h) or [1],
    )
    adapter.engine = SimpleNamespace(tokenizer=None, max_length=64, decide=decide)
    case = InferenceCase.model_validate({
        "case_id": "example", "state": "state",
        "questions": [{"id": "pick", "task": "choice", "instructions": "Choose",
                       "options": [{"id": "a", "description": "Same"},
                                   {"id": "b", "description": "Same"}]}],
    })
    prediction = adapter.predict(case)[0]
    assert prediction.probabilities == {"a": 0.4, "b": 0.6}
    assert seen == ["The correct answer is: option_0: Same"] * 2 + [
        "The correct answer is: option_1: Same",
    ] * 2
