"""Export and synchronize model folders without rewriting measurement provenance."""

import hashlib
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal, Self
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pydantic import Field, model_validator

from .data import Record, Result, load_benchmark, read_json, safe_id, validate_result, write_json


class ModelMetadata(Record):
    model_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*__[A-Za-z0-9][A-Za-z0-9._-]*$")
    display_name: str = Field(min_length=1)
    short_name: str = Field(min_length=1)
    url: str | None = None
    hf_url: str | None = None
    total_params: int | None = Field(default=None, ge=0, strict=True)
    active_params: int | None = Field(default=None, ge=0, strict=True)
    parameter_count_method: Literal["non_lookup_parameters_v1"] | None = None

    @model_validator(mode="after")
    def check_metadata(self) -> Self:
        for url in (self.url, self.hf_url):
            if url is not None:
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError("Model links must be HTTP(S) URLs")
        if self.active_params is not None:
            if self.parameter_count_method != "non_lookup_parameters_v1":
                raise ValueError("Active parameters require non_lookup_parameters_v1")
            if self.total_params is not None and self.active_params > self.total_params:
                raise ValueError("Active parameters exceed total parameters")
        return self


def result_files(paths: list[Path]) -> list[Path]:
    return sorted(
        {
            f
            for p in paths
            for f in (p.rglob("*") if p.is_dir() else [p])
            if f.is_file()
            and f.name != "metadata.json"
            and (f.name.endswith(".json") or f.name.endswith(".json.xz"))
        }
    )


def dataset_source(result: Result) -> dict[str, str]:
    source = result.environment.get("dataset_source", {})
    if not isinstance(source, dict) or not isinstance(source.get("repo_id"), str):
        raise TypeError("Published results require a dataset repo ID and exact revision")
    if not isinstance(source.get("revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", source["revision"]
    ):
        raise ValueError("Published results require a full dataset commit SHA")
    return {key: source[key] for key in ("repo_id", "revision")}


def dataset_cache_key(source: dict[str, str]) -> str:
    return hashlib.sha256(f"{source['repo_id']}@{source['revision']}".encode()).hexdigest()


def result_data_root(root: Path, result: Result, *, download: bool = False) -> Path:
    """Use the recorded release, materializing only the requested subset if needed."""
    source = dataset_source(result)
    from .dataset_source import source_receipts

    for current in source_receipts(root):
        subset_name = Path(result.benchmark.dataset).name
        if all(current.get(k) == v for k, v in source.items()) and (
            "subsets" not in current or subset_name in current["subsets"]
        ):
            return root
    target = root / "result-datasets" / dataset_cache_key(source)
    benchmark = result.benchmark
    subset = benchmark.dataset.removeprefix("datasets/")
    safe_id(subset)
    if benchmark.dataset != f"datasets/{subset}":
        raise ValueError("Invalid benchmark dataset path")
    destination = target / "datasets" / subset
    if destination.exists():
        return target
    if not download:
        raise ValueError(f"Dataset revision not installed: {source}; run sync-results first")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target / ".lock")):
        if destination.exists():
            return target

        def fetch(name: str) -> Path:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid Hub data path")
            return Path(
                hf_hub_download(
                    repo_id=source["repo_id"],
                    revision=source["revision"],
                    repo_type="dataset",
                    filename=name,
                )
            )

        checksums = read_json(fetch("checksums.json"))

        def checked(name: str) -> Path:
            file = fetch(name)
            if checksums.get(name) != hashlib.sha256(file.read_bytes()).hexdigest():
                raise ValueError(f"Hub checksum mismatch: {name}")
            return file

        manifest = read_json(checked("training-manifest.json"))
        entries = [
            e
            for e in manifest["evaluation"]
            if e["dataset"] == subset and e["split"] == benchmark.split
        ]
        if len(entries) != 1:
            raise ValueError("Benchmark is not active in its recorded dataset revision")
        # The active manifest controls membership; all-data supplies file locations.
        all_entries = read_json(checked("all-data-manifest.json"))["evaluation"]
        entry = next(
            e for e in all_entries if e["dataset"] == subset and e["split"] == benchmark.split
        )
        files = [str(checked(f)) for f in entry["data_files"]]
        dataset = Dataset(pa.concat_tables([pq.read_table(f) for f in files]))
        if len(dataset) != entry["cases"]:
            raise ValueError("Hub subset case count mismatch")
        with tempfile.TemporaryDirectory(dir=target, prefix=".subset-") as temporary:
            stage = Path(temporary) / subset
            DatasetDict({benchmark.split: dataset}).save_to_disk(str(stage))
            stage.rename(destination)
    return target


def validate_published(root: Path, result: Result, *, download: bool = False) -> None:
    current = load_benchmark(root, result.benchmark.id)
    # Counts and targets may vary by release; benchmark meaning must stay comparable.
    for key in ("id", "task", "dataset", "split", "primary_metric"):
        if getattr(current, key) != getattr(result.benchmark, key):
            raise ValueError(f"Incompatible benchmark: {result.benchmark.id}")
    validate_result(result_data_root(root, result, download=download), result, published=True)


def export_results(root: Path, paths: list[Path], metadata_path: Path, output: Path) -> Path:
    metadata = ModelMetadata.model_validate(read_json(metadata_path))
    files = result_files(paths)
    if not files:
        raise ValueError("No result files found")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / metadata.model_id
    # Stage one benchmark at a time so large submissions do not retain every prediction.
    with tempfile.TemporaryDirectory(dir=output, prefix=".export-") as temporary:
        stage = Path(temporary)
        for file in files:
            result = Result.model_validate(read_json(file))
            validate_published(root, result)
            staged = stage / f"{result.benchmark.id}.json.xz"
            if staged.exists() and read_json(staged) != result.model_dump():
                raise ValueError(
                    f"Multiple submitted results for {result.benchmark.id}; select one"
                )
            write_json(staged, result.model_dump())
        write_json(stage / "metadata.json", metadata.model_dump())
        # All validation completes before any existing benchmark is replaced.
        with FileLock(str(output / f".{metadata.model_id}.lock")):
            if destination.is_symlink():
                raise ValueError("Model destination must not be a symlink")
            destination.mkdir(exist_ok=True)
            for file in stage.iterdir():
                file.replace(destination / file.name)
    return destination


def validate_repository(root: Path, repository: Path, *, download: bool = False) -> int:
    count = 0
    for folder in sorted(repository.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        metadata = ModelMetadata.model_validate(read_json(folder / "metadata.json"))
        if metadata.model_id != folder.name:
            raise ValueError("Model directory/metadata ID mismatch")
        files = list(folder.iterdir())
        for file in files:
            if file.name == "metadata.json":
                continue
            if not file.is_file() or not file.name.endswith(".json.xz"):
                raise ValueError(f"Unexpected submission file: {file}")
            result = Result.model_validate(read_json(file))
            if file.name != f"{safe_id(result.benchmark.id)}.json.xz":
                raise ValueError("Benchmark filename/ID mismatch")
            validate_published(root, result, download=download)
            count += 1
    if not count:
        raise ValueError("No published results found")
    return count


def sync_results(root: Path, repo_id: str, revision: str, output: Path) -> Path:
    """Install an immutable verified snapshot using an atomic directory symlink switch."""
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.parent / f".{output.name}.lock")):
        if output.exists() and not output.is_symlink():
            raise ValueError("Sync destination must be absent or a managed snapshot symlink")
        sha = HfApi().dataset_info(repo_id, revision=revision).sha
        if sha is None:
            raise ValueError("Could not resolve results revision")
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=sha,
                allow_patterns=["*/metadata.json", "*/*.json.xz"],
                max_workers=4,
            )
        )
        count = validate_repository(root, snapshot, download=True)
        # Materialize cache symlinks as regular files for the viewer's strict walker.
        storage = output.parent / f".{output.name}-snapshots"
        storage.mkdir(exist_ok=True)
        generation = Path(tempfile.mkdtemp(prefix=f"{sha}-", dir=storage))
        try:
            for folder in snapshot.iterdir():
                if folder.is_dir() and not folder.name.startswith("."):
                    shutil.copytree(folder, generation / folder.name)
        except BaseException:
            shutil.rmtree(generation)
            raise
        # Readers resolve the link once per load; previous generations remain usable.
        link = output.parent / f".{output.name}.next"
        link.unlink(missing_ok=True)
        try:
            link.symlink_to(generation, target_is_directory=True)
            link.replace(output)
        finally:
            link.unlink(missing_ok=True)
        print(f"Validated {count} results from {repo_id} @ {sha}; restart the viewer")
    return output
