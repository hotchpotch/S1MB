"""Offline coverage for source selection and guarded display publication."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location("publish_display", Path(__file__).parents[2] / "viewer/scripts/publish-display.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def test_selects_only_original_measurements_and_requires_metadata():
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename=name, size=100) for name in
        ["viewer-summary.json", "README.md", "example__model/test.json.xz", "example__model/metadata.json"]])
    assert publication.result_paths(info) == ["example__model/test.json.xz", "example__model/metadata.json"]
    info.siblings.pop()
    with pytest.raises(ValueError, match="Missing"):
        publication.result_paths(info)
    info.siblings.append(SimpleNamespace(rfilename="nested/invalid/path.json.xz", size=1))
    with pytest.raises(ValueError, match="Unexpected"):
        publication.result_paths(info)


def test_parent_guard_and_single_artifact_upload(tmp_path):
    api = Mock()
    (tmp_path / "viewer-summary.json").write_text('{"payload":{"benchmarks":[],"results":[]}}')
    publication.publish(api, "a" * 40, tmp_path / "viewer-summary.json")
    args = api.create_commit.call_args.kwargs
    assert args["parent_commit"] == "a" * 40
    assert args["repo_id"] == "hotchpotch/s1mb-result"
    assert [op.path_in_repo for op in args["operations"]] == ["viewer-summary.json"]


def test_inventory_rejects_missing_results_even_on_first_publication():
    candidate = {"payload": {"definitions": [0], "benchmarks": [{"id": "test"}],
        "models": [{"run_id": "example__model"}], "results": [{"model": 0, "benchmark": 0}]}}
    publication.check_inventory(["example__model/test.json.xz"], candidate)
    with pytest.raises(ValueError, match="inventory"):
        publication.check_inventory(["example__model/test.json.xz", "missing/test.json.xz"], candidate)


def test_japanese_measurements_publish_to_the_shared_results_dataset(tmp_path):
    import json

    api = Mock()
    output = tmp_path / "viewer-summary.json"
    output.write_text(json.dumps({"payload": {"benchmarks": [{"id": "jnli-choice"}], "results": [{"benchmark": 0}]}}))
    publication.publish(api, "a" * 40, output)
    args = api.create_commit.call_args.kwargs
    assert args["repo_id"] == "hotchpotch/s1mb-result"
    assert args["parent_commit"] == "a" * 40
    assert [op.path_in_repo for op in args["operations"]] == ["viewer-summary.json"]
    api.dataset_info.assert_not_called()
