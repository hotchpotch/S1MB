"""Shared evaluator/viewer examples for imbalance and degenerate baselines."""

import json
from pathlib import Path

import pytest

from s1mb.data import Case, Prediction
from s1mb.metrics import calculate

FIXTURES = json.loads((Path(__file__).parent / "fixtures/diagnostics.json").read_text())


def expand(fixture):
    cases, predictions = [], []
    for sample in fixture["samples"]:
        for _ in range(sample["repeat"]):
            case_id = str(len(cases))
            options = [
                dict(o, description=o["id"]) for o in sample.get("options", fixture["options"])
            ]
            cases.append(
                Case.model_validate(
                    {
                        "case_id": case_id,
                        "group_id": case_id,
                        "language": "en",
                        "state": {},
                        "questions": [
                            {
                                "id": "q",
                                "task": fixture["task"],
                                "instructions": "Judge.",
                                "options": options,
                            }
                        ],
                        "targets": {"q": {"kind": "soft", "probabilities": sample["target"]}},
                        "provenance": {},
                    }
                )
            )
            predictions.append(
                Prediction(case_id=case_id, question_id="q", probabilities=sample["prediction"])
            )
    return cases, predictions


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_shared_scoring_examples(fixture):
    cases, predictions = expand(fixture)
    metrics = calculate(cases, predictions, fixture["task"])
    for key, expected in fixture["expected"].items():
        assert (
            metrics[key] == pytest.approx(expected)
            if expected is not None
            else metrics[key] is None
        )


def test_partial_run_does_not_fit_baseline_to_successful_subset():
    fixture = next(f for f in FIXTURES if f["name"] == "majority_choice")
    cases, predictions = expand(fixture)
    metrics = calculate(cases, predictions[-1:], "choice")
    assert metrics["fixed_answer_accuracy_baseline"] == 0.8
    assert metrics["baseline_adjusted_skill"] == pytest.approx(-4)
    assert metrics["baseline_adjusted_score"] == 0
    assert calculate(cases, [], "choice")["baseline_adjusted_score"] is None


@pytest.mark.dataset
def test_result_diagnostics_reject_tampering(tmp_path):
    from s1mb.adapters.dummy import DummyAdapter
    from s1mb.data import DATA_DIR, load_benchmark, validate_result
    from s1mb.runner import evaluate

    result = evaluate(
        DummyAdapter(),
        DATA_DIR,
        load_benchmark(DATA_DIR, "arc-choice-test-v1"),
        tmp_path,
        "test",
        limit=2,
    )
    validate_result(DATA_DIR, result)
    result.metrics["fixed_answer_accuracy_baseline"] = 0.12345
    with pytest.raises(ValueError, match="recomputed"):
        validate_result(DATA_DIR, result)
