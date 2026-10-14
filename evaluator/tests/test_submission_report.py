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


def test_pr_template_keeps_mixed_provenance_and_author_explanation(tmp_path):
    """Do not collapse distinct revisions or leak per-example hashes into PR text."""
    import lzma

    from s1mb.submission_template import render_template

    folder = tmp_path / "example__model"
    folder.mkdir()
    for revision in ("old", "new"):
        record = {
            "run_id": revision,
            "benchmark": benchmark(revision).model_dump(),
            "model": {"id": "example/model", "adapter": "test", "revision": revision},
            "created_at": "2026-10-13",
            "evaluator_version": "test",
            "provenance": "measured",
            "status": "complete",
            "counts": {"cases": 1, "expected": 1, "succeeded": 1, "failed": 0},
            "metrics": {},
            "elapsed_seconds": 1,
            "environment": {
                "dataset_source": {"revision": revision},
                "input_hashes": {"private-id": "hash"},
            },
            "predictions": [],
        }
        (folder / f"{revision}.json.xz").write_bytes(lzma.compress(json.dumps(record).encode()))
    report = render_template(tmp_path, "english-v1", "My assessment.")
    assert "| choice | 2 | 2 | 2 | 0 |" in report
    assert '"revision": "old"' in report and '"revision": "new"' in report
    assert "dataset_source" in report
    assert "private-id" not in report and "input_hashes" not in report
    assert "My assessment." in report
    assert "Replace this section" not in report
    assert "Replace this section" in render_template(tmp_path, "english-v1", None)


@pytest.mark.parametrize("category", ["english-v1", "japanese-v1"])
@pytest.mark.parametrize("explanation", [None, "My assessment."])
def test_pr_template_always_requests_training_domain_disclosure(tmp_path, category, explanation):
    from s1mb.submission_template import render_template

    report = render_template(tmp_path, category, explanation)
    assert "## Training source and domain disclosure" in report
    assert "For English v1 and Japanese v1" in report
    assert "bekko-system-one-dataset-v0" in report and "saihai-v1-dataset" in report
    assert "other datasets covering the same tasks or source domains" in report
    assert "`training_exposure`" in report
    assert "Missing disclosure does not mean a domain was unseen" in report
    assert "not an explanation or prediction of high scores" in report
    if explanation:
        assert explanation in report


def test_abstention_disclosure_is_scoped_by_task():
    model = ModelSummary(
        ModelMetadata(model_id="test__model", display_name="Model", short_name="M"), False
    )
    model.measurements = {
        "choice": Measurement(True, measurement().metrics, abstained=2),
        "noul": Measurement(True, measurement().metrics, abstained=3),
        "outside": Measurement(False, measurement().metrics, abstained=99, failed=99),
    }
    report = render_report([benchmark("choice"), benchmark("noul", "noul")], [model])
    assert "| Model | 3 | 2 | 0 |" in report
    assert "2/2 complete" in report
    assert "99 failed" not in report
    assert "penalties reduce the reported score" in report
    assert "no additional flat deduction" in report
