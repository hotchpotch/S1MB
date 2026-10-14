"""Validate packaged submissions and render reviewable leaderboard comparisons."""

import argparse
import html
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from .data import DATA_DIR, Benchmark, Result, Task, load_benchmark, load_category, read_json
from .diagnostics import EPSILON
from .result_repository import ModelMetadata, validate_repository

TASKS: tuple[Task, ...] = ("noul", "choice", "score")


@dataclass
class Measurement:
    complete: bool
    metrics: dict[str, float | None]
    abstained: int = 0
    failed: int = 0

    @classmethod
    def from_result(cls, result: Result) -> "Measurement":
        return cls(
            result.status == "complete"
            and result.provenance == "measured"
            and result.counts.failed == 0,
            result.metrics,
            result.counts.abstained,
            result.counts.failed,
        )


@dataclass
class ModelSummary:
    metadata: ModelMetadata
    reference: bool
    measurements: dict[str, Measurement] = field(default_factory=dict)
    parameter_counts: set[tuple[int | None, int | None]] = field(default_factory=set)


def eligible(task: Task, metrics: dict[str, float | None]) -> bool:
    """Use validated baselines from each result's own dataset revision."""
    key = {
        "noul": "positive_prevalence",
        "choice": "fixed_answer_accuracy_baseline",
        "score": "constant_mae_baseline",
    }[task]
    value = metrics.get(key)
    if value is None:
        raise ValueError(f"Missing baseline: {key}")
    if task == "noul":
        return EPSILON < value < 1 - EPSILON
    return value < 1 - EPSILON if task == "choice" else value > EPSILON


def task_score(required: list[Benchmark], measurements: dict[str, Measurement]) -> float | None:
    """Require complete coverage before excluding target-degenerate benchmarks."""
    if not required or any(
        b.id not in measurements or not measurements[b.id].complete for b in required
    ):
        return None
    scores = []
    for benchmark in required:
        metrics = measurements[benchmark.id].metrics
        if not eligible(benchmark.task, metrics):
            continue
        value = metrics.get("baseline_adjusted_score")
        if value is None:
            return None
        if not 0 <= value <= 1:
            raise ValueError("Adjusted scores must already be clipped per benchmark")
        scores.append(value)
    return 100 * mean(scores) if scores else None


def load_models(root: Path, repository: Path, *, reference: bool, download: bool):
    count = validate_repository(root, repository, download=download)
    print(f"Validated {count} results in {repository}", file=sys.stderr)
    models = []
    for folder in sorted(repository.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        metadata = ModelMetadata.model_validate(read_json(folder / "metadata.json"))
        model = ModelSummary(metadata, reference)
        # Keep summaries only; never retain an entire repository's predictions.
        for path in sorted(folder.glob("*.json.xz")):
            result = Result.model_validate(read_json(path))
            model.measurements[result.benchmark.id] = Measurement.from_result(result)
            model.parameter_counts.add((result.model.total_params, result.model.active_params))
        models.append(model)
    return models


def cell(value: str) -> str:
    return html.escape(value).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")


def display_parameters(model: ModelSummary) -> tuple[int | None, int | None]:
    """Match viewer metadata precedence, including explicitly unknown counts."""
    metadata = model.metadata
    if metadata.total_params is not None or metadata.active_params is not None:
        return metadata.total_params, metadata.active_params
    if len(model.parameter_counts) > 1:
        raise ValueError(f"Inconsistent parameter counts: {metadata.model_id}; set metadata counts")
    return next(iter(model.parameter_counts), (None, None))


def format_parameters(value: int | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    return f"{value:,}"


def render_report(benchmarks: list[Benchmark], models: list[ModelSummary]) -> str:
    lines = [
        "## Leaderboard comparison",
        "",
        (
            "| Model | Total Params | Active Params | noul | choice | score "
            "| gen-noul | gen-choice | gen-score |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    incomplete = []
    abstention_rows = []
    for model in models:
        scores = []
        for general in (False, True):
            for task in TASKS:
                required = [
                    b
                    for b in benchmarks
                    if b.task == task
                    and (not general or b.dataset.startswith("datasets/s1mb-generalization-"))
                ]
                value = task_score(required, model.measurements)
                scores.append("N/A" if value is None else f"{value:.2f}")
        name = cell(model.metadata.display_name) + (" (reference)" if model.reference else "")
        total, active = map(format_parameters, display_parameters(model))
        lines.append(f"| {name} | {total} | {active} | " + " | ".join(scores) + " |")
        complete = sum(
            b.id in model.measurements and model.measurements[b.id].complete for b in benchmarks
        )
        scoped_failures = sum(
            model.measurements[b.id].failed for b in benchmarks if b.id in model.measurements
        )
        task_abstentions = [
            sum(
                model.measurements[b.id].abstained
                for b in benchmarks
                if b.task == task and b.id in model.measurements
            )
            for task in TASKS
        ]
        if any(task_abstentions):
            abstention_rows.append(f"| {name} | " + " | ".join(map(str, task_abstentions)) + " |")
        if complete != len(benchmarks) or scoped_failures:
            incomplete.append(
                f"{name}: {complete}/{len(benchmarks)} complete, {scoped_failures} failed decisions"
            )
    lines += [
        "",
        (
            "Scores: baseline-adjusted 0–100, higher is better; equal benchmark weights. "
            "Noul uses balanced accuracy. Gen is the Diverse/Contextual subset of each task. "
            "Static Active Params excludes embedding weights, including tied output weights (`embedding_excluded_parameters_v1`); historic results retain their recorded method. "
            "N/A means unknown parameters or unavailable scores."
        ),
        "",
    ]
    if incomplete:
        lines.append("Coverage exceptions: " + "; ".join(incomplete) + ".")
    else:
        lines.append(
            f"All {len(models)} rows: **{len(benchmarks)}/{len(benchmarks)} complete, "
            "zero failed decisions**."
        )
    if abstention_rows:
        lines += [
            "",
            "### Unanswerable judgments counted as incorrect",
            "",
            "| Model | Noul | Choice | Score |",
            "| --- | ---: | ---: | ---: |",
            *abstention_rows,
            "",
            "Abstentions remain in scoring denominators and receive zero correct credit or maximum normalized error. Operational failures remain incomplete.",
        ]
    lines.append("Validated against each result's recorded dataset revision.")
    return "\n".join(lines) + "\n"


def main(*, template: bool = False) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Packaged submission repository, containing model folders",
    )
    parser.add_argument(
        "--reference-dir",
        action="append",
        default=[],
        type=Path,
        help="Separate packaged reference repository; repeatable",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--category", default="english-v1")
    parser.add_argument("--download-datasets", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Reject incomplete submitted models; references may be partial",
    )
    parser.add_argument("--output", type=Path, help="Markdown file; stdout if omitted")
    if template:
        parser.add_argument(
            "--explanation-file", type=Path, help="Author-written English explanation to append"
        )
    args = parser.parse_args()
    try:
        category = load_category(args.data_dir, args.category)
        benchmarks = [load_benchmark(args.data_dir, bid) for bid in category.benchmarks]
        models = load_models(
            args.data_dir, args.results_dir, reference=False, download=args.download_datasets
        )
        if args.require_complete:
            for model in models:
                if any(
                    b.id not in model.measurements or not model.measurements[b.id].complete
                    for b in benchmarks
                ):
                    raise ValueError(f"Incomplete submitted model: {model.metadata.model_id}")
        for reference in args.reference_dir:
            models.extend(
                load_models(
                    args.data_dir, reference, reference=True, download=args.download_datasets
                )
            )
        report = render_report(benchmarks, models)
        if template:
            from .submission_template import render_template

            explanation = (
                args.explanation_file.read_text(encoding="utf-8") if args.explanation_file else None
            )
            report += "\n" + render_template(args.results_dir, args.category, explanation)
        if args.output is None:
            print(report, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".report-") as temp:
                staged = Path(temp) / "report.md"
                staged.write_text(report, encoding="utf-8")
                staged.replace(args.output)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Submission report failed: {error}\n")


if __name__ == "__main__":
    main()
