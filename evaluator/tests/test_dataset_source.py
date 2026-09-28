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
