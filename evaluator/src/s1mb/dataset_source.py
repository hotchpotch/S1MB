"""Materialize the latest Hub evaluation release for the evaluator and JS viewer."""

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_from_disk
from filelock import FileLock
from huggingface_hub import HfApi, snapshot_download

from .data import DATA_DIR, load_benchmark, load_cases, load_category

SOURCE_CONFIG = DATA_DIR.parent / "dataset-source.json"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def dataset_session(root: Path, *, offline: bool = False):
    """Keep a CLI run and dataset refresh from replacing each other's inputs."""
    root = root.resolve()
    scratch = root.parent.parent / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    with FileLock(str(scratch / f"dataset-{identity}.lock")):
        if not offline:
            ensure_dataset(root)
        yield


def ensure_dataset(root: Path, source_config: Path = SOURCE_CONFIG):
    """Resolve latest once and materialize only if the installed revision differs."""
    source = read(source_config)
    requested_revision = source.get("revision", "main")
    resolved = HfApi().dataset_info(source["repo_id"], revision=requested_revision).sha
    if resolved is None:
        raise ValueError("Could not resolve the dataset revision")
    source["revision"] = resolved
    installed_path = root / "datasets/hub-source.json"
    if installed_path.exists():
        installed = read(installed_path)
        if (
            installed.get("repo_id") == source["repo_id"]
            and installed.get("revision") == resolved
            and installed.get("subsets")
            and all(
                (root / "datasets" / name / "dataset_dict.json").is_file()
                for name in installed["subsets"]
            )
        ):
            print(f"Dataset: {source['repo_id']} @ {resolved} (cached)", flush=True)
            return installed
    snapshot = Path(snapshot_download(**source))
    checksums = read(snapshot / "checksums.json")
    for name, expected in checksums.items():
        path = (snapshot / name).resolve()
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Hub checksum mismatch: {name}")
    scratch = root.parent.parent / "tmp"
    scratch.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="hub-dataset-", dir=scratch))
    stage = work / "data"
    stage.mkdir()
    target = stage / "datasets"
    shutil.copytree(snapshot, target)
    # Keep published Parquet and manifests intact; Arrow is a local materialization.
    entries = read(snapshot / "all-data-manifest.json")
    active = read(snapshot / "training-manifest.json")["evaluation"]
    # All-data manifest retains quarantined subsets, never added to categories.
    rows = entries["evaluation"]
    counts = {}
    for entry in rows:
        name, split = entry["dataset"], entry["split"]
        if Path(name).name != name or Path(split).name != split:
            raise ValueError("Invalid subset/split name")
        table = pa.concat_tables([pq.read_table(snapshot / f) for f in entry["data_files"]])
        dataset = Dataset(table)
        if len(dataset) != entry["cases"]:
            raise ValueError(f"Case count mismatch: {name}")
        local = root / "datasets" / name
        if local.exists():
            previous = load_from_disk(str(local))[split]
            if previous.to_list() != dataset.to_list():
                raise ValueError(
                    f"Existing dataset differs: {name}; review before updating results"
                )
        DatasetDict({split: dataset}).save_to_disk(str(target / name))
        if load_from_disk(str(target / name))[split].to_list() != dataset.to_list():
            raise ValueError(f"Arrow roundtrip mismatch: {name}")
        counts[name] = len(dataset)
    shutil.copytree(root / "benchmarks", stage / "benchmarks")
    shutil.copytree(root / "categories", stage / "categories")
    active_names = {entry["dataset"] for entry in active}
    category = load_category(stage, "english-v1")
    used = set()
    for name in category.benchmarks:
        benchmark = load_benchmark(stage, name)
        used.add(Path(benchmark.dataset).name)
        load_cases(stage, benchmark)
    if used != active_names:
        raise ValueError("Benchmark definitions do not match the Hub active membership")
    (target / "hub-source.json").write_text(
        json.dumps(
            {
                **source,
                "requested_revision": requested_revision,
                "materialization": "datasets.save_to_disk",
                "subsets": counts,
            },
            indent=2,
        )
        + "\n"
    )
    backup = work / "previous-datasets"
    destination = root / "datasets"
    if destination.exists():
        destination.rename(backup)
    try:
        target.rename(destination)
    except BaseException:
        if backup.exists():
            backup.rename(destination)
        raise
    print(f"Installed {len(counts)} subsets from {source['revision']}; backup: {backup}")

    return read(destination / "hub-source.json")
