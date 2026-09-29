"""Run each benchmark once and preserve predictions, including failed decisions."""

import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .adapters.base import ModelAdapter
from .data import (
    Benchmark,
    Case,
    Counts,
    Prediction,
    Result,
    check_probabilities,
    load_cases,
    read_json,
    validate_result,
    write_json,
)
from .metrics import calculate


def predict_cases(adapter: ModelAdapter, cases: list[Case]):
    """Use bounded batches when supported; isolate failed batches by case."""
    batch_predict = getattr(adapter, "predict_batch", None)
    size = getattr(adapter, "case_batch_size", 1) if batch_predict else 1
    for offset in range(0, len(cases), size):
        group = cases[offset : offset + size]
        if batch_predict:
            try:
                outputs = batch_predict([case.inference() for case in group])
                if len(outputs) != len(group):
                    raise ValueError("Adapter returned missing or duplicate cases")
            except Exception as exc:  # noqa: BLE001 - isolate failures
                print(f"  batch failed: {type(exc).__name__}; retrying cases", flush=True)
                outputs = None
            if outputs is not None:
                yield from zip(group, outputs, strict=True)
                continue
        for case in group:
            try:
                output = adapter.predict(case.inference())
            except Exception as exc:  # noqa: BLE001 - preserve per-case failure handling
                yield case, exc
            else:
                yield case, output


def evaluate(
    adapter: ModelAdapter,
    root: Path,
    benchmark: Benchmark,
    output: Path,
    run_id: str,
    limit: int | None = None,
) -> Result:
    destination = output / run_id / f"{benchmark.id}.json"
    if destination.exists():
        raise ValueError(f"Result already exists: {destination}; use a new run ID")
    source_path = root / "datasets/hub-source.json"
    dataset_source = read_json(source_path) if source_path.exists() else None
    cases = load_cases(root, benchmark)
    selected = cases[:limit] if limit is not None else cases
    predictions = []
    usage_before = dict(getattr(adapter, "usage", {}))
    start = time.perf_counter()
    for i, (case, current) in enumerate(predict_cases(adapter, selected)):
        try:
            if isinstance(current, Exception):
                raise current
            questions = {q.id: q for q in case.questions}
            if len(current) != len(questions) or {p.question_id for p in current} != set(questions):
                raise ValueError("Adapter returned missing or duplicate questions")
            for pred in current:
                if pred.case_id != case.case_id or pred.probabilities is None:
                    raise ValueError("Adapter returned an invalid case ID or missing probabilities")
                check_probabilities(
                    pred.probabilities, [o.id for o in questions[pred.question_id].options]
                )
            predictions.extend(current)
        except Exception as exc:  # noqa: BLE001 - preserve a failed case without losing the run
            # Exception messages from third-party clients may contain sensitive request data.
            error = type(exc).__name__
            print(f"  failed {case.case_id}: {error}", flush=True)
            predictions.extend(
                Prediction(case_id=case.case_id, question_id=q.id, error=error)
                for q in case.questions
            )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(selected)} cases", flush=True)
    failed = sum(p.error is not None for p in predictions)
    model = adapter.metadata()
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_hashes": {c.case_id: c.provenance["input_hash"] for c in cases},
    }
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        try:
            environment["evaluator_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    if dataset_source is not None:
        environment["dataset_source"] = {
            key: dataset_source[key] for key in ("repo_id", "revision")
        }
    if usage_before:
        environment["api_usage"] = {
            key: value - usage_before[key] for key, value in getattr(adapter, "usage", {}).items()
        }
    result = Result(
        run_id=run_id,
        benchmark=benchmark,
        model=model,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        evaluator_version=version("s1mb"),
        provenance="demo" if model.adapter == "dummy" else "measured",
        status="complete" if len(selected) == len(cases) and not failed else "partial",
        counts=Counts(
            cases=len(selected),
            expected=benchmark.decision_count,
            succeeded=len(predictions) - failed,
            failed=failed,
        ),
        metrics=calculate(cases, predictions, benchmark.task),
        elapsed_seconds=time.perf_counter() - start,
        environment=environment,
        predictions=predictions,
    )
    validate_result(root, result)
    write_json(destination, result.model_dump())
    return result
