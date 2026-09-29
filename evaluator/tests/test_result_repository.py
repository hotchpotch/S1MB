"""Public submission tests use synthetic Arrow data and mocked Hub transport."""

import copy
import hashlib
import lzma
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from s1mb.adapters.dummy import DummyAdapter
from s1mb.data import Benchmark, Task, read_json, write_json
from s1mb.result_repository import (
    ModelMetadata,
    dataset_cache_key,
    export_results,
    result_data_root,
    sync_results,
    validate_repository,
)
from s1mb.runner import evaluate


@pytest.fixture
def submission(tmp_path):
    root = tmp_path / "data"
    row = read_json(Path(__file__).parent / "fixtures/system_one.json")
    source = {"repo_id": "test/dataset", "revision": "a" * 40}
    DatasetDict({"test": Dataset.from_list([row])}).save_to_disk(str(root / "datasets/sample"))
    write_json(root / "datasets/hub-source.json", source)
    benchmarks = []
    tasks: list[tuple[Task, str]] = [
        ("choice", "target_mass_at_prediction"),
        ("noul", "binary_brier"),
    ]
    for task, metric in tasks:
        benchmark = Benchmark(
            id=f"sample-{task}",
            task=task,
            dataset="datasets/sample",
            split="test",
            case_count=1,
            decision_count=1,
            primary_metric=metric,
        )
        write_json(root / "benchmarks" / f"{benchmark.id}.json", benchmark.model_dump())
        benchmarks.append(benchmark)
    metadata = tmp_path / "metadata.json"
    write_json(
        metadata,
        {
            "model_id": "test__model_v1",
            "display_name": "Model v1",
            "short_name": "Model",
            "active_params": 10,
            "parameter_count_method": "non_lookup_parameters_v1",
        },
    )
    first = evaluate(DummyAdapter(), root, benchmarks[0], tmp_path / "raw", "first")
    return root, metadata, first, benchmarks


def test_export_append_replace_and_validate(submission, tmp_path):
    root, metadata, first, benchmarks = submission
    output = tmp_path / "public"
    destination = export_results(root, [tmp_path / "raw/first"], metadata, output)
    file = destination / f"{first.benchmark.id}.json.xz"
    assert read_json(file)["run_id"] == "first"
    assert read_json(file)["environment"]["dataset_source"]["revision"] == "a" * 40
    evaluate(DummyAdapter(), root, benchmarks[1], tmp_path / "raw", "second")
    export_results(root, [tmp_path / "raw/second"], metadata, output)
    assert validate_repository(root, output) == 2
    replacement = first.model_copy(update={"run_id": "replacement"})
    write_json(tmp_path / "replacement.json", replacement.model_dump())
    export_results(root, [tmp_path / "replacement.json"], metadata, output)
    assert read_json(file)["run_id"] == "replacement"
    assert validate_repository(root, output) == 2
    bad = replacement.model_dump()
    bad["metrics"][first.benchmark.primary_metric] = 0.123
    write_json(tmp_path / "bad.json", bad)
    with pytest.raises(ValueError, match="metrics"):
        export_results(root, [tmp_path / "bad.json"], metadata, output)
    assert read_json(file)["run_id"] == "replacement"


def test_mixed_revision_uses_own_inputs(submission, tmp_path):
    root, metadata, first, _ = submission
    old_source = {"repo_id": "test/dataset", "revision": "b" * 40}
    cache = root / "result-datasets" / dataset_cache_key(old_source)
    row = read_json(Path(__file__).parent / "fixtures/system_one.json")
    row["input_hash"] = "older-input"
    DatasetDict({"test": Dataset.from_list([row])}).save_to_disk(str(cache / "datasets/sample"))
    write_json(cache / "datasets/hub-source.json", old_source)
    write_json(cache / "benchmarks" / f"{first.benchmark.id}.json", first.benchmark.model_dump())
    old = evaluate(DummyAdapter(), cache, first.benchmark, tmp_path / "old", "old")
    assert result_data_root(root, old) == cache
    export_results(root, [tmp_path / "old"], metadata, tmp_path / "public")
    assert validate_repository(root, tmp_path / "public") == 1
    old.environment["dataset_source"]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="not installed"):
        result_data_root(root, old)


def test_sync_is_pinned_and_failure_preserves_previous(submission, tmp_path, monkeypatch):
    root, metadata, first, _ = submission
    remote = tmp_path / "remote"
    folder = export_results(root, [tmp_path / "raw/first"], metadata, remote)
    monkeypatch.setattr(
        "s1mb.result_repository.HfApi",
        lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="d" * 40)),
    )
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(remote)

    monkeypatch.setattr("s1mb.result_repository.snapshot_download", download)
    output = root / "hub-results"
    sync_results(root, "test/results", "refs/pr/1", output)
    previous = output.resolve()
    assert calls[0]["revision"] == "d" * 40
    assert validate_repository(root, output) == 1
    bad = copy.deepcopy(first.model_dump())
    bad["environment"]["input_hashes"] = {}
    write_json(folder / f"{first.benchmark.id}.json.xz", bad)
    with pytest.raises(ValueError, match="hashes"):
        sync_results(root, "test/results", "main", output)
    assert output.resolve() == previous
    assert validate_repository(root, output) == 1


def test_metadata_and_compressed_json_reject_invalid_data(tmp_path):
    with pytest.raises(ValueError):
        ModelMetadata(model_id="../escape", display_name="x", short_name="x")
    with pytest.raises(ValueError):
        ModelMetadata(
            model_id="test__x", display_name="x", short_name="x", url="javascript:alert(1)"
        )
    file = tmp_path / "bad.json.xz"
    file.write_bytes(lzma.compress(b'{"a":1,"a":2}'))
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(file)
    file.write_bytes(b"not xz")
    with pytest.raises((lzma.LZMAError, ValueError)):
        read_json(file)


def test_download_recorded_subset_and_verify_checksum(submission, tmp_path, monkeypatch):
    root, _, result, _ = submission
    remote = tmp_path / "dataset-hub"
    remote.mkdir()
    row = read_json(Path(__file__).parent / "fixtures/system_one.json")
    Dataset.from_list([row]).to_parquet(str(remote / "sample.parquet"))
    entry = {"dataset": "sample", "split": "test", "cases": 1, "data_files": ["sample.parquet"]}
    write_json(remote / "training-manifest.json", {"evaluation": [entry]})
    write_json(remote / "all-data-manifest.json", {"evaluation": [entry]})
    write_json(
        remote / "checksums.json",
        {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in remote.iterdir()},
    )
    result.environment["dataset_source"]["revision"] = "c" * 40
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return str(remote / kwargs["filename"])

    monkeypatch.setattr("s1mb.result_repository.hf_hub_download", fetch)
    target = result_data_root(root, result, download=True)
    assert (target / "datasets/sample/test/state.json").is_file()
    assert all(call["revision"] == "c" * 40 for call in calls)
    # Another uncached release must fail before installing altered data.
    result.environment["dataset_source"]["revision"] = "d" * 40
    (remote / "sample.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        result_data_root(root, result, download=True)
    key = dataset_cache_key(result.environment["dataset_source"])
    assert not (root / "result-datasets" / key / "datasets/sample").exists()


def test_compressed_size_limit_and_truncated_stream(tmp_path, monkeypatch):
    file = tmp_path / "result.json.xz"
    monkeypatch.setattr("s1mb.data.MAX_JSON_BYTES", 1024)
    file.write_bytes(lzma.compress(b'"' + b"x" * 2048 + b'"'))
    with pytest.raises(ValueError, match="exceeds"):
        read_json(file)
    file.write_bytes(lzma.compress(b'{"ok":true}')[:-4])
    with pytest.raises(ValueError, match="Truncated"):
        read_json(file)
