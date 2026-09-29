"""Offline checks for deployment boundaries and startup failures."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


deployment = module("deploy")
bootstrap = module("bootstrap")
IMAGE = "ghcr.io/example/s1mb@sha256:" + "a" * 64
SHA = "b" * 40


@pytest.mark.parametrize("image", ["ghcr.io/example/s1mb:latest", IMAGE + "\nRUN false"])
def test_reject_mutable_or_injected_image(image):
    with pytest.raises(ValueError):
        deployment.payload(image, SHA)


@pytest.mark.parametrize("private,sdk", [(False, "docker"), (True, "gradio")])
def test_wrong_space_is_never_modified(private, sdk):
    api = Mock()
    api.space_info.return_value = SimpleNamespace(private=private, sdk=sdk, sha=SHA)
    with pytest.raises(ValueError):
        deployment.deploy(api, "example/S1MB-leaderboard", deployment.payload(IMAGE, SHA), SHA)
    api.create_commit.assert_not_called()


def test_deploy_preserves_history_and_checks_parent():
    api = Mock()
    api.space_info.return_value = SimpleNamespace(private=True, sdk="docker", sha=SHA)
    api.get_space_variables.return_value = {"S1MB_HF_RESULTS_REPO": object()}
    api.create_commit.return_value = SimpleNamespace(oid="c" * 40)
    assert deployment.deploy(api, "example/S1MB-leaderboard",
                             deployment.payload(IMAGE, SHA), SHA) == "c" * 40
    kwargs = api.create_commit.call_args.kwargs
    assert kwargs["parent_commit"] == SHA
    assert {op.path_in_repo for op in kwargs["operations"]} == {"README.md", "Dockerfile"}
    assert kwargs["operations"][1].path_or_fileobj == f"# Source commit: {SHA}\nFROM {IMAGE}\n".encode()


def test_bootstrap_requires_configuration_before_network(monkeypatch, tmp_path):
    session = Mock()
    monkeypatch.setattr(bootstrap, "dataset_session", session)
    with pytest.raises(ValueError):
        bootstrap.prepare(tmp_path, {})
    session.assert_not_called()


def test_prepare_current_and_recorded_revisions(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(bootstrap, "dataset_session",
                        lambda root: (order.append(("current", root)) or nullcontext()))
    sync = Mock(side_effect=lambda *args: order.append(("results", args)))
    monkeypatch.setattr(bootstrap, "sync_results", sync)
    bootstrap.prepare(tmp_path, {"S1MB_HF_RESULTS_REPO": "example/results",
                                 "S1MB_HF_RESULTS_REVISION": SHA})
    assert order[0] == ("current", tmp_path)
    sync.assert_called_once_with(tmp_path, "example/results", SHA, tmp_path / "hub-results")


def test_failed_sync_never_starts_server(monkeypatch):
    monkeypatch.setattr(bootstrap.os, "chdir", Mock())
    monkeypatch.setattr(bootstrap, "prepare", Mock(side_effect=ValueError("invalid data")))
    execute = Mock()
    monkeypatch.setattr(bootstrap.os, "execvpe", execute)
    with pytest.raises(ValueError, match="invalid data"):
        bootstrap.main([])
    execute.assert_not_called()


def test_startup_passes_same_cli_settings_to_sync_and_viewer(monkeypatch):
    monkeypatch.setattr(bootstrap.os, "chdir", Mock())
    monkeypatch.setenv("S1MB_HF_RESULTS_REPO", "example/from-environment")
    monkeypatch.setenv("S1MB_HF_RESULTS_REVISION", "main")
    prepare = Mock()
    execute = Mock()
    monkeypatch.setattr(bootstrap, "prepare", prepare)
    monkeypatch.setattr(bootstrap.os, "execvpe", execute)
    bootstrap.main(["--results-repo", "example/from-cli", "--results-revision", SHA])
    prepared_env = prepare.call_args.args[1]
    viewer_env = execute.call_args.args[2]
    for env in [prepared_env, viewer_env]:
        assert env["S1MB_HF_RESULTS_REPO"] == "example/from-cli"
        assert env["S1MB_HF_RESULTS_REVISION"] == SHA


def test_environment_is_used_without_cli_overrides():
    env = {"S1MB_HF_RESULTS_REPO": "example/results", "S1MB_HF_RESULTS_REVISION": SHA}
    assert bootstrap.runtime_environment([], env) == env


def test_old_running_image_is_not_success(monkeypatch):
    api = Mock(token=False)
    api.space_info.return_value = SimpleNamespace(private=True, sha=SHA)
    api.get_space_runtime.return_value = SimpleNamespace(stage="RUNNING", raw={"sha": "c" * 40})
    monkeypatch.setattr(deployment.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    monkeypatch.setattr(deployment.time, "sleep", Mock())
    request = Mock()
    monkeypatch.setattr(deployment.httpx, "get", request)
    with pytest.raises(TimeoutError):
        deployment.wait_ready(api, "example/S1MB-leaderboard", SHA, 1)
    request.assert_not_called()


def test_private_space_readiness(monkeypatch):
    api = Mock(token="synthetic-token")
    api.space_info.return_value = SimpleNamespace(private=True, sha=SHA,
                                                  host="https://example-s1mb-leaderboard.hf.space")
    api.get_space_runtime.return_value = SimpleNamespace(stage="RUNNING", raw={"sha": SHA})
    request = Mock(return_value=SimpleNamespace(status_code=200, text="S1MB"))
    monkeypatch.setattr(deployment.httpx, "get", request)
    deployment.wait_ready(api, "example/S1MB-leaderboard", SHA, 1)
    assert request.call_args.kwargs["follow_redirects"] is False
    assert request.call_args.kwargs["headers"]["authorization"] == "Bearer synthetic-token"
