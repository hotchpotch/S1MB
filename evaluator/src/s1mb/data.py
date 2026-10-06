"""Shared file formats and validation for fixed-sample benchmarks."""

from __future__ import annotations

import json
import lzma
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_JSON_BYTES = 64 * 1024 * 1024

Task = Literal["choice", "noul", "score"]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
if not DATA_DIR.is_dir():
    # Installed wheels use the checkout's data, selected from the working directory.
    DATA_DIR = Path.cwd() / "data"
METRICS = {
    "choice": "target_mass_at_prediction",
    "noul": "binary_brier",
    "score": "normalized_expected_score_mae",
}


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Option(Record):
    id: str
    description: str
    description_json: str | None = None
    value: float | None = None


class Question(Record):
    id: str
    task: Task
    instructions: str = Field(min_length=1)
    instructions_json: str | None = None
    system_prompt: str = ""
    options: list[Option] = Field(min_length=2)

    @model_validator(mode="after")
    def check_options(self) -> Self:
        ids = [o.id for o in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate option IDs")
        if self.task == "noul" and set(ids) != {"false", "true"}:
            raise ValueError("Noul requires false/true option IDs")
        if self.task == "score":
            values = [o.value for o in self.options]
            if any(v is None for v in values) or len(set(values)) < 2:
                raise ValueError("Score requires a nonconstant numeric scale")
        return self


def check_probabilities(probabilities: dict[str, float], ids: list[str]) -> None:
    if set(probabilities) != set(ids):
        raise ValueError("Probability keys must match the option IDs exactly")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5):
        raise ValueError("Probabilities must sum to one")


class Target(Record):
    kind: str
    probabilities: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceCase(Record):
    case_id: str
    state: Any
    state_json: str | None = None
    questions: list[Question]


class Case(InferenceCase):
    group_id: str
    language: str
    targets: dict[str, Target]
    provenance: dict[str, Any]

    @model_validator(mode="after")
    def check_targets(self) -> Self:
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids) or set(ids) != set(self.targets):
            raise ValueError("Question and target IDs must match without duplicates")
        for q in self.questions:
            check_probabilities(self.targets[q.id].probabilities, [o.id for o in q.options])
        return self

    def inference(self) -> InferenceCase:
        return InferenceCase(
            case_id=self.case_id,
            state=self.state,
            state_json=self.state_json,
            questions=self.questions,
        )


class Benchmark(Record):
    id: str
    task: Task
    dataset: str
    split: Literal["test"]
    case_count: int = Field(gt=0)
    decision_count: int = Field(gt=0)
    primary_metric: str

    @model_validator(mode="after")
    def check_metric(self) -> Self:
        safe_id(self.id)
        if self.primary_metric != METRICS[self.task]:
            raise ValueError("Unexpected primary metric")
        return self


class Category(Record):
    id: str
    name: str
    description: str
    benchmarks: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_ids(self) -> Self:
        safe_id(self.id)
        if len(set(self.benchmarks)) != len(self.benchmarks):
            raise ValueError("Duplicate benchmarks in category")
        for name in self.benchmarks:
            safe_id(name)
        return self


class Prediction(Record):
    case_id: str
    question_id: str
    probabilities: dict[str, float] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        if (self.probabilities is None) == (self.error is None):
            raise ValueError("A prediction needs exactly one of probabilities or error")
        return self


class ModelInfo(Record):
    id: str
    adapter: str
    revision: str
    settings: dict[str, Any] = Field(default_factory=dict)
    total_params: int | None = Field(default=None, ge=0, strict=True)
    active_params: int | None = Field(default=None, ge=0, strict=True)
    parameter_count_method: Literal["non_lookup_parameters_v1"] | None = None

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        if self.active_params is not None and (
            self.total_params is None or self.active_params > self.total_params
        ):
            raise ValueError("Active parameters require total_params >= active_params")
        if (self.active_params is not None) != (self.parameter_count_method is not None):
            raise ValueError("Active parameters require a counting method")
        return self


class Counts(Record):
    cases: int = Field(ge=0)
    expected: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)


class Result(Record):
    format_version: Literal[1] = 1
    run_id: str
    benchmark: Benchmark
    model: ModelInfo
    created_at: str
    evaluator_version: str
    provenance: Literal["measured", "demo"]
    status: Literal["complete", "partial"]
    counts: Counts
    metrics: dict[str, float | None]
    elapsed_seconds: float = Field(ge=0)
    environment: dict[str, Any]
    predictions: list[Prediction]


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def read_json(path: Path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    # Bound both the compressed input and the decompressed output before parsing.
    with path.open("rb") as stream:
        payload = stream.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds 64 MiB: {path}")
    if path.name.endswith(".json.xz"):
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=128 * 1024 * 1024)
        payload = decoder.decompress(payload, max_length=MAX_JSON_BYTES + 1)
        if len(payload) > MAX_JSON_BYTES:
            raise ValueError(f"JSON exceeds 64 MiB: {path}")
        if not decoder.eof or decoder.unused_data:
            raise ValueError(f"Truncated XZ stream or trailing data: {path}")
    return json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds 64 MiB: {path}")
    if path.name.endswith(".json.xz"):
        payload = lzma.compress(payload)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_benchmark(root: Path, name: str) -> Benchmark:
    result = Benchmark.model_validate(read_json(root / "benchmarks" / f"{safe_id(name)}.json"))
    if result.id != name:
        raise ValueError("Benchmark filename/ID mismatch")
    return result


def load_category(root: Path, name: str) -> Category:
    result = Category.model_validate(read_json(root / "categories" / f"{safe_id(name)}.json"))
    if result.id != name:
        raise ValueError("Category filename/ID mismatch")
    return result


def load_cases(root: Path, benchmark: Benchmark) -> list[Case]:
    path = (root / benchmark.dataset).resolve()
    if not path.is_relative_to((root / "datasets").resolve()):
        raise ValueError("Dataset path escapes the datasets directory")
    from .hf_data import load_hf_cases

    cases = []
    for case in load_hf_cases(path, benchmark.split):
        questions = [q for q in case.questions if q.task == benchmark.task]
        if questions:
            cases.append(
                case.model_copy(
                    update={
                        "questions": questions,
                        "targets": {q.id: case.targets[q.id] for q in questions},
                    }
                )
            )
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    if (
        len(cases) != benchmark.case_count
        or sum(len(c.questions) for c in cases) != benchmark.decision_count
    ):
        raise ValueError("Dataset counts differ from the benchmark definition")
    return cases


def validate_result(root: Path, result: Result, *, published: bool = False) -> None:
    from .metrics import calculate

    safe_id(result.run_id)
    if result.model.adapter == "dummy" and result.provenance != "demo":
        raise ValueError("Dummy adapter results must be marked as demo")
    benchmark = result.benchmark if published else load_benchmark(root, result.benchmark.id)
    if benchmark != result.benchmark:
        raise ValueError("Result benchmark differs from the fixed definition")
    cases = load_cases(root, benchmark)
    input_hashes = result.environment.get("input_hashes")
    expected_hashes = {c.case_id: c.provenance["input_hash"] for c in cases}
    if input_hashes != expected_hashes:
        raise ValueError("Result input hashes differ from the dataset or are missing")
    index = {(c.case_id, q.id): q for c in cases for q in c.questions}
    seen = set()
    for pred in result.predictions:
        key = (pred.case_id, pred.question_id)
        if key not in index or key in seen:
            raise ValueError("Unknown or duplicate prediction ID")
        seen.add(key)
        if pred.probabilities is not None:
            check_probabilities(pred.probabilities, [o.id for o in index[key].options])
    succeeded = sum(p.probabilities is not None for p in result.predictions)
    expected_counts = Counts(
        cases=len({p.case_id for p in result.predictions}),
        expected=len(index),
        succeeded=succeeded,
        failed=len(result.predictions) - succeeded,
    )
    if result.counts != expected_counts:
        raise ValueError("Result counts do not match predictions")
    complete = len(seen) == len(index) and expected_counts.failed == 0
    if (result.status == "complete") != complete:
        raise ValueError("Incorrect completion status")
    expected = calculate(cases, result.predictions, benchmark.task)
    if set(result.metrics) != set(expected):
        raise ValueError("Stored metrics differ from recomputed metrics")
    for key, actual in result.metrics.items():
        value = expected[key]
        if value is None or actual is None:
            if value != actual:
                raise ValueError("Stored metrics differ from recomputed metrics")
        elif not math.isclose(value, actual, abs_tol=1e-8):
            raise ValueError("Stored metrics differ from recomputed metrics")
