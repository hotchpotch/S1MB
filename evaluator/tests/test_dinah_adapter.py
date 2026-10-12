"""Dinah semantics and probability validation without model downloads."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.dinah import DinahAdapter, dinah_questions


def test_numeric_score_and_structured_anonymous_choice():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    case.questions[2].options.reverse()
    questions = dinah_questions(case)
    assert questions["choose"]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["criteria"] == [
        {"value": 30.0, "description": "High"},
        {"value": 10.0, "description": "Low"},
    ]
    assert "gold-must-not-be-sent" not in str(questions)


def test_single_question_calls_and_alignment():
    model = DinahAdapter.__new__(DinahAdapter)

    def predict(rows, batch_size):
        assert len(rows) == batch_size == 1
        row = rows[0]
        assert row["state"] == cases().state
        if row["type"] == "noul":
            return [{"type": "noul", "noul": 0.7}]
        keys = list(row["criteria"]) if row["type"] == "choice" else ["0", "1"]
        return [{"type": row["type"], "probabilities": dict(zip(keys, [0.2, 0.8]))}]

    model.engine = SimpleNamespace(predict=predict)
    results = model.predict(cases())
    assert results[0].probabilities == {"x": 0.2, "y": 0.8}
    assert results[1].probabilities["true"] == 0.7
    assert results[2].probabilities == {"low": 0.2, "high": 0.8}


@pytest.mark.parametrize("results", [[], [{"type": "choice", "probabilities": {"x": 1}}]])
def test_invalid_outputs_rejected(results):
    model = DinahAdapter.__new__(DinahAdapter)
    model.engine = SimpleNamespace(predict=lambda *args, **kwargs: results)
    with pytest.raises(ValueError):
        model.predict(cases())


@pytest.mark.parametrize("device,limit", [("cpu", None), ("cuda:0", 0), ("cuda:0", 8193)])
def test_invalid_settings_before_download(device, limit):
    with pytest.raises(ValueError):
        DinahAdapter("unused", "main", device, context_limit=limit)


def test_native_overflow_propagates_without_retry_or_shortening():
    model = DinahAdapter.__new__(DinahAdapter)

    def predict(rows, batch_size):
        raise ValueError("request needs 8193 tokens (no truncation)")

    model.engine = SimpleNamespace(predict=predict)
    with pytest.raises(ValueError, match="no truncation"):
        model.predict(cases())
