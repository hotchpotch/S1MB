"""Resolve current Hub revisions once; never silently fall back to stale inputs."""

import json
from types import SimpleNamespace

import pytest

from s1mb import cli, dataset_source


def test_cached_revision_checks_hub_without_downloading(tmp_path, monkeypatch):
    config = tmp_path / "source.json"
    config.write_text(
        json.dumps({"repo_id": "owner/data", "revision": "main", "repo_type": "dataset"})
    )
    root = tmp_path / "data"
    subset = root / "datasets/example"
    subset.mkdir(parents=True)
    (subset / "dataset_dict.json").write_text("{}")
    receipt = {"repo_id": "owner/data", "revision": "a" * 40, "subsets": {"example": 1}}
    (root / "datasets/hub-source.json").write_text(json.dumps(receipt))
    requested = []

    def info(repo, revision):
        requested.append((repo, revision))
        return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(dataset_source, "HfApi", lambda: SimpleNamespace(dataset_info=info))

    def unexpected(**kwargs):
        pytest.fail("Cached data must not be downloaded")

    monkeypatch.setattr(dataset_source, "snapshot_download", unexpected)
    assert dataset_source.ensure_dataset(root, config) == receipt
    assert requested == [("owner/data", "main")]

    def unavailable(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(dataset_source, "HfApi", lambda: SimpleNamespace(dataset_info=unavailable))
    with pytest.raises(ConnectionError):
        dataset_source.ensure_dataset(root, config)
    assert json.loads((root / "datasets/hub-source.json").read_text()) == receipt


def test_run_refreshes_once_and_offline_is_explicit(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(dataset_source, "ensure_dataset", lambda root: events.append("refresh"))
    monkeypatch.setattr(cli, "execute", lambda args, parser: events.append("evaluate"))
    command = ["s1mb", "--data-dir", str(tmp_path / "data"), "run", "--adapter", "dummy"]
    monkeypatch.setattr("sys.argv", command)
    cli.main()
    assert events == ["refresh", "evaluate"]
    events.clear()
    monkeypatch.setattr("sys.argv", [*command, "--offline-dataset"])
    cli.main()
    assert events == ["evaluate"]


def test_category_source_cache_and_provenance_are_isolated(tmp_path, monkeypatch):
    config = tmp_path / "source.json"
    config.write_text(
        json.dumps(
            {
                "repo_id": "owner/english",
                "revision": "main",
                "repo_type": "dataset",
                "category_sources": {
                    "japanese-v1": {
                        "repo_id": "owner/japanese",
                        "revision": "main",
                        "repo_type": "dataset",
                    }
                },
            }
        )
    )
    root = tmp_path / "data"
    for subset in ("english", "japanese"):
        folder = root / "datasets" / subset
        folder.mkdir(parents=True)
        (folder / "dataset_dict.json").write_text("{}")
    english = {"repo_id": "owner/english", "revision": "e" * 40, "subsets": {"english": 1}}
    japanese = {"repo_id": "owner/japanese", "revision": "j" * 40, "subsets": {"japanese": 1}}
    (root / "datasets/hub-source.json").write_text(json.dumps(english))
    receipts = root / "datasets/hub-sources"
    receipts.mkdir()
    (receipts / "japanese-v1.json").write_text(json.dumps(japanese))
    requested = []

    def info(repo, revision):
        requested.append(repo)
        return SimpleNamespace(sha="j" * 40 if repo.endswith("japanese") else "e" * 40)

    monkeypatch.setattr(dataset_source, "HfApi", lambda: SimpleNamespace(dataset_info=info))
    monkeypatch.setattr(
        dataset_source, "snapshot_download", lambda **kw: pytest.fail("Unexpected download")
    )
    assert dataset_source.ensure_dataset(root, config, category="japanese-v1") == japanese
    assert dataset_source.ensure_dataset(root, config) == english
    assert requested == ["owner/japanese", "owner/english"]
    assert dataset_source.source_for_subset(root, "japanese") == japanese
    assert dataset_source.source_for_subset(root, "english") == english
    with pytest.raises(ValueError, match="No dataset source"):
        dataset_source.ensure_dataset(root, config, category="unknown")
    assert len(requested) == 2


def test_japanese_run_selects_category_source(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        dataset_source, "ensure_dataset", lambda root, **kw: events.append(kw["category"])
    )
    monkeypatch.setattr(cli, "execute", lambda args, parser: events.append("evaluate"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s1mb",
            "--data-dir",
            str(tmp_path / "data"),
            "run",
            "--adapter",
            "dummy",
            "--category",
            "japanese-v1",
        ],
    )
    cli.main()
    assert events == ["japanese-v1", "evaluate"]


def test_vision_release_roundtrip_and_failed_refresh_is_atomic(tmp_path, monkeypatch):
    """A private visual release restores exact cases/blobs without touching text data."""
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    from s1mb.data import load_benchmark, load_cases, write_json

    root = tmp_path / "installed"
    root.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = b"synthetic image bytes; no inference"
    sha = hashlib.sha256(blob).hexdigest()
    case = {"case_id": "c", "group_id": "g", "language": "en", "state": "Find red",
            "questions": [{"id": "q", "task": "box", "instructions": "Find the rectangle"}],
            "targets": {"q": {"kind": "box", "box": {"left": 0, "top": 0, "right": 1, "bottom": 1}}},
            "images": {"image_1": {"path": "../image-cache/" + sha, "sha256": sha}},
            "provenance": {"input_hash": "test"}}
    definition = {"id": "visual", "dataset": "datasets/visual", "split": "test", "task": "box",
                  "case_count": 1, "decision_count": 1, "primary_metric": "mean_iou"}
    write_json(root / "benchmarks/visual.json", definition)
    write_json(root / "categories/vision-v1.json", {"id": "vision-v1", "name": "Vision",
               "description": "Synthetic", "benchmarks": ["visual"]})
    pq.write_table(pa.table({"case_json": [json.dumps(case)]}), snapshot / "cases.parquet")
    pq.write_table(pa.table({"sha256": [sha], "bytes": [blob]}), snapshot / "assets.parquet")
    entry = {"dataset": "visual", "split": "test", "cases": 1, "data_files": ["cases.parquet"],
             "format": "s1mb-vision-case-v1"}
    for name in ("training-manifest.json", "all-data-manifest.json"):
        write_json(snapshot / name, {"evaluation": [entry]})
    write_json(snapshot / "vision-assets.json", {"files": ["assets.parquet"], "count": 1})
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()}
    write_json(snapshot / "checksums.json", checksums)
    config = tmp_path / "source.json"
    write_json(config, {"category_sources": {"vision-v1": {"repo_id": "test/vision", "revision": "main", "repo_type": "dataset"}}})
    monkeypatch.setattr(dataset_source, "HfApi", lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="a" * 40)))
    monkeypatch.setattr(dataset_source, "snapshot_download", lambda **kwargs: str(snapshot))
    receipt = dataset_source.ensure_dataset(root, config, category="vision-v1")
    loaded = load_cases(root, load_benchmark(root, "visual"))
    target_box = loaded[0].targets["q"].box
    assert target_box is not None
    assert target_box.right == 1
    assert (root / "datasets/image-cache" / sha).read_bytes() == blob
    assert receipt["repo_id"] == "test/vision"
    (snapshot / "cases.parquet").write_bytes(b"corrupt")
    monkeypatch.setattr(dataset_source, "HfApi", lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="b" * 40)))
    with pytest.raises(ValueError, match="checksum"):
        dataset_source.ensure_dataset(root, config, category="vision-v1")
    assert json.loads((root / "datasets/hub-sources/vision-v1.json").read_text()) == receipt
