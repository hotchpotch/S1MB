"""Abstentions are scored outcomes; transport errors and missing work are not."""

import pytest

from s1mb.data import Prediction


def test_outcomes_are_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        Prediction(case_id="c", question_id="q", probabilities={"a": 1}, abstention="refusal")
    with pytest.raises(ValueError, match="exactly one"):
        Prediction(case_id="c", question_id="q", error="HTTP 400", abstention="refusal")


@pytest.mark.parametrize("operational_error", [False, True])
def test_runner_preserves_other_questions_and_counts_abstention(
    monkeypatch, tmp_path, operational_error
):
    from s1mb import runner
    from s1mb.data import Benchmark, Case, ModelInfo

    case = Case.model_validate(
        {
            "case_id": "c",
            "group_id": "c",
            "language": "en",
            "state": {},
            "questions": [
                {
                    "id": q,
                    "task": "noul",
                    "instructions": "Judge.",
                    "options": [
                        {"id": "false", "description": "No"},
                        {"id": "true", "description": "Yes"},
                    ],
                }
                for q in ["a", "b"]
            ],
            "targets": {
                q: {"kind": "hard", "probabilities": {"false": 0, "true": 1}} for q in ["a", "b"]
            },
            "provenance": {"input_hash": "h"},
        }
    )
    benchmark = Benchmark(
        id="test",
        task="noul",
        dataset="datasets/test",
        split="test",
        case_count=1,
        decision_count=2,
        primary_metric="binary_brier",
    )
    monkeypatch.setattr(runner, "load_cases", lambda *args: [case])
    monkeypatch.setattr(runner, "source_for_subset", lambda *args: None)
    monkeypatch.setattr(runner, "validate_result", lambda *args: None)

    class Adapter:
        def predict(self, case):
            return [
                Prediction(case_id="c", question_id="a", error="HTTPError")
                if operational_error
                else Prediction(case_id="c", question_id="a", abstention="refusal"),
                Prediction(case_id="c", question_id="b", probabilities={"false": 0, "true": 1}),
            ]

        def close(self):
            pass

        def metadata(self):
            return ModelInfo(id="synthetic", adapter="test", revision="test")

    result = runner.evaluate(Adapter(), tmp_path, benchmark, tmp_path, "fresh")
    from s1mb import data

    monkeypatch.setattr(data, "load_cases", lambda *args: [case])
    monkeypatch.setattr(data, "load_benchmark", lambda *args: benchmark)
    data.validate_result(tmp_path, result)
    assert result.predictions[1].probabilities == {"false": 0, "true": 1}
    if operational_error:
        assert result.status == "partial"
        assert (result.counts.succeeded, result.counts.abstained, result.counts.failed) == (1, 0, 1)
        assert result.metrics["binary_brier"] == 0
        assert "abstention_scoring" not in result.environment
    else:
        assert result.status == "complete"
        assert (result.counts.succeeded, result.counts.abstained, result.counts.failed) == (1, 1, 0)
        assert result.metrics["binary_brier"] == 0.5
        assert result.environment["abstention_scoring"] == "incorrect-max-error-v1"
        with pytest.raises(ValueError, match="explicit scoring policy"):
            data.validate_result(
                tmp_path, result.model_copy(update={"environment": {"input_hashes": {"c": "h"}}})
            )
        with pytest.raises(ValueError, match="counts do not match"):
            data.validate_result(
                tmp_path,
                result.model_copy(
                    update={"counts": result.counts.model_copy(update={"abstained": 0})}
                ),
            )
