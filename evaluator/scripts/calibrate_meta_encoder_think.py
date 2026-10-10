"""Select and apply one global temperature to persisted MetaEncoder-think results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from s1mb.adapters.meta_encoder_think import recalibrate_probabilities
from s1mb.data import Result, load_cases, read_json, validate_result, write_json
from s1mb.metrics import calculate
from s1mb.result_repository import result_files


def log_grid(low: float, high: float, count: int) -> list[float]:
    start, stop = math.log(low), math.log(high)
    return [math.exp(start + (stop - start) * index / (count - 1)) for index in range(count)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-temperature", type=float, default=0.03)
    parser.add_argument("--minimum", type=float, default=1e-4)
    parser.add_argument("--maximum", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=161)
    parser.add_argument("--refine-size", type=int, default=401)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    if args.grid_size < 3 or args.refine_size < 3 or not 0 < args.minimum < args.maximum:
        parser.error("invalid temperature grid")

    inputs = result_files(args.paths)
    parsed = [Result.model_validate(read_json(path)) for path in inputs]
    if not parsed:
        parser.error("no result files found")
    if len({result.benchmark.id for result in parsed}) != len(parsed):
        parser.error("each benchmark must appear exactly once")
    cases = {result.benchmark.id: load_cases(args.data_dir, result.benchmark) for result in parsed}
    original_choice = {}
    for result in parsed:
        validate_result(args.data_dir, result)
        if not math.isclose(
            result.model.settings.get("temperature", math.nan),
            args.source_temperature,
            abs_tol=1e-15,
        ):
            parser.error("saved source temperature does not match --source-temperature")
        index = {
            (case.case_id, question.id): question
            for case in cases[result.benchmark.id]
            for question in case.questions
        }
        for prediction in result.predictions:
            if prediction.probabilities is None:
                parser.error("calibration requires complete predictions")
            question = index[prediction.case_id, prediction.question_id]
            ids = [option.id for option in question.options]
            if list(prediction.probabilities) != ids:
                parser.error("probability keys must preserve declared option order")
            if result.benchmark.task == "choice":
                original_choice[(result.benchmark.id, prediction.case_id, prediction.question_id)] = max(
                    ids, key=prediction.probabilities.__getitem__
                )

    cache = {}

    def evaluate(temperature: float):
        if temperature in cache:
            return cache[temperature]
        task_scores = defaultdict(list)
        metrics = {}
        valid = True
        for result in parsed:
            converted = []
            for prediction in result.predictions:
                assert prediction.probabilities is not None
                try:
                    probabilities = recalibrate_probabilities(
                        prediction.probabilities, args.source_temperature, temperature
                    )
                except ValueError:
                    valid = False
                    break
                if result.benchmark.task == "choice":
                    key = (result.benchmark.id, prediction.case_id, prediction.question_id)
                    if max(probabilities, key=probabilities.__getitem__) != original_choice[key]:
                        valid = False
                        break
                converted.append(prediction.model_copy(update={"probabilities": probabilities}))
            if not valid:
                break
            current = calculate(cases[result.benchmark.id], converted, result.benchmark.task)
            metrics[result.benchmark.id] = current
            task_scores[result.benchmark.task].append(current["baseline_adjusted_score"])
        scores = (
            {task: math.fsum(values) / len(values) for task, values in task_scores.items()}
            if valid and set(task_scores) == {"choice", "noul", "score"}
            else {}
        )
        row = {
            "temperature": temperature,
            "valid": bool(scores),
            "task_scores": scores,
            "task_avg": math.fsum(scores.values()) / 3 if scores else None,
            "metrics": metrics,
        }
        cache[temperature] = row
        return row

    broad = sorted(set(log_grid(args.minimum, args.maximum, args.grid_size) + [args.source_temperature]))
    for temperature in broad:
        evaluate(temperature)
    broad_best = max((cache[value] for value in broad if cache[value]["valid"]), key=lambda row: row["task_avg"])
    best_index = broad.index(broad_best["temperature"])
    lower = broad[max(0, best_index - 1)]
    upper = broad[min(len(broad) - 1, best_index + 1)]
    for temperature in log_grid(lower, upper, args.refine_size):
        evaluate(temperature)
    winner = max((row for row in cache.values() if row["valid"]), key=lambda row: row["task_avg"])

    args.output.mkdir(parents=True)
    results_output = args.output / "results"
    audit_output = args.output / "audit"
    results_output.mkdir()
    audit_output.mkdir()
    now = datetime.now(UTC).isoformat()
    for source_path, result in zip(inputs, parsed, strict=True):
        payload = read_json(source_path)
        payload["predictions"] = [
            prediction.model_copy(
                update={
                    "probabilities": recalibrate_probabilities(
                        prediction.probabilities,
                        args.source_temperature,
                        winner["temperature"],
                    )
                }
            ).model_dump()
            for prediction in result.predictions
        ]
        payload["metrics"] = winner["metrics"][result.benchmark.id]
        payload["run_id"] = f"{result.run_id}-global-cal"
        payload["created_at"] = now
        payload["model"]["settings"]["source_temperature"] = args.source_temperature
        payload["model"]["settings"]["temperature"] = winner["temperature"]
        payload["model"]["settings"]["temperature_calibration"] = {
            "method": "persisted-probability-power-softmax-fp64-v1",
            "selection": "one-global-temperature-maximizing-equal-type-task-avg",
            "posthoc_full_evaluation_calibration": True,
            "raw_logits_recovered": False,
        }
        payload["environment"]["probability_postprocessing"] = {
            "source_run_id": result.run_id,
            "source_file_sha256": sha256(source_path),
            "source_temperature": args.source_temperature,
            "effective_temperature": winner["temperature"],
            "gpu_rerun": False,
        }
        write_json(results_output / source_path.name, payload)
        validate_result(args.data_dir, Result.model_validate(payload))

    report = []
    for row in sorted(cache.values(), key=lambda item: item["temperature"]):
        report.append({key: value for key, value in row.items() if key != "metrics"})
    write_json(audit_output / "temperature-sweep.json", report)
    write_json(
        audit_output / "selected-temperature.json",
        {
            "source_temperature": args.source_temperature,
            "effective_temperature": winner["temperature"],
            "task_scores": winner["task_scores"],
            "task_avg": winner["task_avg"],
            "selection_disclosure": (
                "Selected posthoc on the complete evaluation set; not untuned test generalization."
            ),
            "probability_limitation": (
                "Uses persisted probabilities and does not exactly recover raw cosine logits."
            ),
        },
    )
    print(json.dumps(read_json(audit_output / "selected-temperature.json"), indent=2))


if __name__ == "__main__":
    main()
