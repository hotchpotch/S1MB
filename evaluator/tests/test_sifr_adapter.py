"""Explicit Choice mapping preserves Noul definitions and numeric Score levels."""

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.sifr import SifrAdapter, sifr_questions


def test_native_choice_mapping_keeps_complete_typed_criteria():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested":true}'
    case.questions[2].options.reverse()
    questions = sifr_questions(case)
    assert all(q["type"] == "choice" for q in questions.values())
    assert questions["choose"]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["criteria"] == {
        "option_0": {"value": 30.0, "description": "High"},
        "option_1": {"value": 10.0, "description": "Low"},
    }
    assert "gold-must-not-be-sent" not in str(questions)


def test_native_key_probabilities_align_with_original_options():
    adapter = SifrAdapter.__new__(SifrAdapter)

    def engine(state, questions):
        assert state == cases().state
        assert set(questions) == {"decision"}
        question = questions["decision"]
        assert question["type"] == "choice"
        probabilities = dict(zip(question["criteria"], [0.2, 0.8], strict=True))
        return {"answers": {"decision": {"probabilities": probabilities}}}, {}

    adapter.engine = engine
    results = adapter.predict(cases())
    assert results[0].probabilities == {"x": 0.2, "y": 0.8}
    assert results[1].probabilities == {"true": 0.2, "false": 0.8}
    assert results[2].probabilities == {"low": 0.2, "high": 0.8}


def test_misaligned_native_answers_rejected():
    adapter = SifrAdapter.__new__(SifrAdapter)
    adapter.engine = lambda *args: ({"answers": {"other": {}}}, {})
    with pytest.raises(ValueError, match="count"):
        adapter.predict(cases())


def test_native_distribution_validation():
    adapter = SifrAdapter.__new__(SifrAdapter)
    adapter.engine = lambda *args: ({"answers": {"decision": {
        "probabilities": {"option_0": 0.1, "option_1": 0.2}
    }}}, {})
    with pytest.raises(ValueError, match="sum"):
        adapter.predict(cases())


def test_invalid_budget_rejected_before_model_load():
    with pytest.raises(ValueError, match="positive"):
        SifrAdapter("unused", "main", "unused", "cuda:0", context_limit=0)


def test_experimental_cache_path_refused(monkeypatch):
    monkeypatch.setenv("SIFR_KV", "1")
    with pytest.raises(ValueError, match="full-sequence"):
        SifrAdapter("unused", "main", "unused", "cuda:0")
