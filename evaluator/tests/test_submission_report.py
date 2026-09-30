"""PR comparisons retain coverage gates and the viewer's scoring conventions."""

import json
from pathlib import Path

import pytest

from s1mb.data import METRICS, Benchmark, Task
from s1mb.result_repository import ModelMetadata
from s1mb.submission_report import (
    Measurement,
    ModelSummary,
    display_parameters,
    eligible,
    format_parameters,
    render_report,
    task_score,
)


def benchmark(name: str, task: Task = "choice", general: bool = False) -> Benchmark:
    return Benchmark(
        id=name,
        task=task,
        dataset="datasets/s1mb-generalization-test" if general else "datasets/test",
        split="test",
        case_count=1,
        decision_count=1,
        primary_metric=METRICS[task],
    )


def measurement(score=0.5, baseline=0.5, complete=True):
    return Measurement(
        complete,
        {
            "baseline_adjusted_score": score,
            "fixed_answer_accuracy_baseline": baseline,
            "positive_prevalence": baseline,
            "constant_mae_baseline": baseline,
        },
    )


def test_equal_weights_and_ineligible_coverage_gate():
    required = [benchmark("a"), benchmark("b"), benchmark("degenerate")]
    values = {
        "a": measurement(0),
        "b": measurement(0.8),
        "degenerate": measurement(None, baseline=1),
    }
    assert task_score(required, values) == 40
    del values["degenerate"]
    assert task_score(required, values) is None
    values["degenerate"] = measurement(None, baseline=1, complete=False)
    assert task_score(required, values) is None
    values["degenerate"] = measurement(None, baseline=1)
    values["b"] = measurement(None)
    assert task_score(required, values) is None
    assert task_score([], values) is None


def test_reference_baselines_are_not_replaced_by_submission_baselines():
    required = [benchmark("a"), benchmark("b")]
    assert task_score(required, {"a": measurement(None, 1), "b": measurement(0.8)}) == 80
    assert task_score(required, {"a": measurement(0), "b": measurement(0.8)}) == 40


def test_shared_diagnostic_fixtures():
    fixtures = json.loads((Path(__file__).parent / "fixtures/diagnostics.json").read_text())
    checked = 0
    for fixture in fixtures:
        metrics = fixture["expected"]
        baseline = {
            "choice": "fixed_answer_accuracy_baseline",
            "noul": "positive_prevalence",
            "score": "constant_mae_baseline",
        }[fixture["task"]]
        if baseline not in metrics or "baseline_adjusted_score" not in metrics:
            continue
        expected = metrics["baseline_adjusted_score"]
        actual = task_score(
            [benchmark("example", fixture["task"])], {"example": Measurement(True, metrics)}
        )
        assert actual == (pytest.approx(100 * expected) if expected is not None else None)
        checked += 1
    assert checked >= 3


def test_six_columns_general_subset_and_reference_label():
    required = [benchmark("normal"), benchmark("general", general=True)]
    model = ModelSummary(
        ModelMetadata(model_id="test__model", display_name="A|B", short_name="A"), reference=True
    )
    model.measurements = {"normal": measurement(0), "general": measurement(0.8)}
    report = render_report(required, [model])
    assert "| noul | choice | score | gen-noul | gen-choice | gen-score |" in report
    assert "| A&#124;B (reference) | N/A | N/A | N/A | 40.00 | N/A | N/A | 80.00 | N/A |" in report
    model.measurements.pop("normal")
    report = render_report(required, [model])
    assert "| A&#124;B (reference) | N/A | N/A | N/A | N/A | N/A | N/A | 80.00 | N/A |" in report
    assert "1/2 complete" in report


def test_parameter_counts_follow_viewer_metadata_precedence():
    model = ModelSummary(
        ModelMetadata(model_id="test__model", display_name="Model", short_name="Model"), False
    )
    assert display_parameters(model) == (None, None)
    model.parameter_counts.add((321908995, 125298691))
    assert display_parameters(model) == (321908995, 125298691)
    assert format_parameters(321908995) == "321.909M"
    assert format_parameters(4034264576) == "4.034B"
    assert format_parameters(None) == "N/A"
    model.parameter_counts.add((400000000, 200000000))
    with pytest.raises(ValueError, match="Inconsistent parameter counts"):
        display_parameters(model)
    model.metadata.total_params = 500000000
    assert display_parameters(model) == (500000000, None)


def test_no_silent_missing_baseline_or_unclipped_score():
    with pytest.raises(ValueError, match="Missing baseline"):
        eligible("noul", {})
    with pytest.raises(ValueError, match="clipped"):
        task_score([benchmark("a")], {"a": measurement(-0.4)})
