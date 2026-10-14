"""Offline checks for deployment boundaries and startup failures."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


deployment = module("deploy")
IMAGE = "ghcr.io/example/s1mb@sha256:" + "a" * 64
SHA = "b" * 40


@pytest.mark.parametrize("image", ["ghcr.io/example/s1mb:latest", IMAGE + "\nRUN false"])
def test_reject_mutable_or_injected_image(image):
    with pytest.raises(ValueError):
        deployment.payload(image, SHA, [])


@pytest.mark.parametrize("private,sdk", [(False, "gradio"), (True, "gradio")])
def test_wrong_space_is_never_modified(private, sdk):
    api = Mock()
    api.space_info.return_value = SimpleNamespace(private=private, sdk=sdk, sha=SHA)
    with pytest.raises(ValueError):
        deployment.deploy(api, "example/S1MB-leaderboard", deployment.payload(IMAGE, SHA, ["example/model"]), SHA)
    api.create_commit.assert_not_called()


@pytest.mark.parametrize("private", [True, False])
def test_deploy_preserves_history_and_checks_parent(monkeypatch, private):
    monkeypatch.setattr(deployment, "hf_hub_download",
                        Mock(side_effect=deployment.EntryNotFoundError("missing")))
    api = Mock()
    api.space_info.return_value = SimpleNamespace(private=private, sdk="docker", sha=SHA,
        runtime=SimpleNamespace(raw={"volumes": [{"type": "dataset", "source": "hotchpotch/s1mb-result",
            "mountPath": "/mnt/results", "readOnly": True},
            {"type": "bucket", "mountPath": "/mnt/cache", "readOnly": False}]}))
    api.create_commit.return_value = SimpleNamespace(oid="c" * 40)
    assert deployment.deploy(api, "example/S1MB-leaderboard",
                             deployment.payload(IMAGE, SHA, ["example/model"]), SHA) == "c" * 40
    kwargs = api.create_commit.call_args.kwargs
    assert kwargs["parent_commit"] == SHA
    assert {op.path_in_repo for op in kwargs["operations"]} == {"README.md", "Dockerfile", "models.py"}
    operations = {op.path_in_repo: op.path_or_fileobj for op in kwargs["operations"]}
    assert operations["models.py"] == deployment.render_models(["example/model"])
    assert operations["Dockerfile"] == f"# Source commit: {SHA}\nFROM {IMAGE}\n".encode()


def test_old_running_image_is_not_success(monkeypatch):
    api = Mock(token=False)
    api.space_info.return_value = SimpleNamespace(private=True, sha=SHA)
    api.get_space_runtime.return_value = SimpleNamespace(stage="RUNNING", raw={"sha": "c" * 40})
    monkeypatch.setattr(deployment.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    monkeypatch.setattr(deployment.time, "sleep", Mock())
    request = Mock()
    monkeypatch.setattr(deployment.httpx, "stream", request)
    with pytest.raises(TimeoutError):
        deployment.wait_ready(api, "example/S1MB-leaderboard", SHA, 1, private=True)
    request.assert_not_called()


@pytest.mark.parametrize("private", [True, False])
def test_space_readiness(monkeypatch, private):
    api = Mock(token="synthetic-token")
    api.space_info.return_value = SimpleNamespace(private=private, sha=SHA,
                                                  host="https://example-s1mb-leaderboard.hf.space")
    api.get_space_runtime.return_value = SimpleNamespace(stage="RUNNING", raw={"sha": SHA})
    request = MagicMock()
    request.return_value.__enter__.side_effect = [
        deployment.httpx.Response(200, text="Preparing the leaderboard"),
        deployment.httpx.Response(200, json={"version": 1, "payload": {"results": [{}]}}),
    ]
    monkeypatch.setattr(deployment.httpx, "stream", request)
    deployment.wait_ready(api, "example/S1MB-leaderboard", SHA, 1, private=private)
    assert request.call_args.kwargs["follow_redirects"] is False
    if private:
        assert request.call_args.kwargs["headers"]["authorization"] == "Bearer synthetic-token"
    else:
        assert "authorization" not in request.call_args.kwargs["headers"]


@pytest.mark.parametrize("url,expected", [
    ("https://huggingface.co/example/model", "example/model"),
    ("https://huggingface.co/example/model/tree/abcdef/checkpoint", "example/model"),
    ("https://huggingface.co/example/model/resolve/main/config.json", "example/model"),
    ("https://huggingface.co/example/model/blob/main/README.md", "example/model"),
    ("https://huggingface.co/datasets/example/data", None),
    ("https://huggingface.co/spaces/example/app", None),
    ("https://huggingface.co/blog/example/post", None),
    ("https://huggingface.co/example/model/discussions/1", None),
    ("https://huggingface.co.evil.test/example/model", None),
    ("https://example.com/model", None),
])
def test_model_page_normalization(url, expected):
    assert deployment.model_name(url) == expected


def test_models_render_is_stable_and_inert():
    content = deployment.render_models(["z/model", "A/model", "z/model"])
    assert content == deployment.render_models(["A/model", "z/model"])
    import ast
    module = ast.parse(content)
    assert ast.literal_eval(module.body[1].value) == ["A/model", "z/model"]
    assert len(module.body) == 2  # Docstring and literal assignment only.
    with pytest.raises(ValueError):
        deployment.render_models(['example/model"; print("injected")'])


def metadata_snapshot(tmp_path, entries):
    import json
    files = []
    paths = {}
    for index, links in enumerate(entries):
        folder = f"example__model{index}"
        name = f"{folder}/metadata.json"
        data = {"model_id": folder, "display_name": "Synthetic", "short_name": "Synthetic",
                **links}
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(data))
        paths[name] = str(path)
        files.extend([SimpleNamespace(rfilename=name, size=path.stat().st_size),
                      SimpleNamespace(rfilename=f"{folder}/benchmark.json.xz", size=100)])
    api = Mock(token=False)
    api.dataset_info.return_value = SimpleNamespace(sha=SHA, siblings=files)
    download = Mock(side_effect=lambda repo, filename, **kwargs: paths[filename])
    return api, download


def test_published_models_pin_metadata_and_skip_api_models(tmp_path, monkeypatch):
    api, download = metadata_snapshot(tmp_path, [
        {"hf_url": "https://huggingface.co/example/model/tree/main/small"},
        {"hf_url": "https://huggingface.co/example/model/tree/main/large"},
        {"url": "https://huggingface.co/another/model"},
        {"url": "https://example.com/api-model"},
        {"hf_url": None, "url": None},
    ])
    # Metadata without any published benchmark files does not establish model coverage.
    api.dataset_info.return_value.siblings.append(
        SimpleNamespace(rfilename="empty__folder/metadata.json", size=10))
    monkeypatch.setattr(deployment, "hf_hub_download", download)
    assert deployment.published_models(api, "example/results") == ["another/model", "example/model"]
    api.dataset_info.assert_called_once_with("example/results", revision="main", files_metadata=True)
    assert download.call_count == 5
    for call in download.call_args_list:
        assert call.kwargs == {"repo_type": "dataset", "revision": SHA, "token": False}
        assert call.args[1].endswith("/metadata.json")


@pytest.mark.parametrize("failure", ["missing", "oversized", "invalid_link", "invalid_json",
                                     "identity", "download"])
def test_bad_metadata_stops_generation(tmp_path, monkeypatch, failure):
    api, download = metadata_snapshot(tmp_path, [
        {"hf_url": "https://huggingface.co/example/model"},
    ])
    siblings = api.dataset_info.return_value.siblings
    if failure == "missing":
        siblings.pop(0)
    elif failure == "oversized":
        siblings[0].size = 1024 * 1024 + 1
    elif failure in {"invalid_json", "identity", "invalid_link"}:
        import json
        path = tmp_path / "0.json"
        data = json.loads(path.read_text())
        if failure == "identity":
            data["model_id"] = "another__model"
        elif failure == "invalid_link":
            data["hf_url"] = "https://huggingface.co/datasets/example/data"
        path.write_text("{invalid" if failure == "invalid_json" else json.dumps(data))
        siblings[0].size = path.stat().st_size
    else:
        download.side_effect = OSError("offline")
    monkeypatch.setattr(deployment, "hf_hub_download", download)
    with pytest.raises((ValueError, OSError)):
        deployment.published_models(api, "example/results")
    if failure in {"missing", "oversized"}:
        download.assert_not_called()


@pytest.mark.parametrize("changed", [False, True])
def test_only_changed_models_are_deployed(tmp_path, monkeypatch, changed):
    files = deployment.payload(IMAGE, SHA, ["example/new-model"])
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    if changed:
        (tmp_path / "models.py").write_bytes(deployment.render_models(["example/old-model"]))
    api = Mock(token=False)
    info = SimpleNamespace(sha=SHA)
    monkeypatch.setattr(deployment, "check_space", Mock(return_value=info))
    download = Mock(side_effect=lambda repo, name, **kwargs: str(tmp_path / name))
    monkeypatch.setattr(deployment, "hf_hub_download", download)
    api.create_commit.return_value = SimpleNamespace(oid="c" * 40)
    result = deployment.deploy(api, "example/S1MB-leaderboard", files, SHA)
    if changed:
        assert result == "c" * 40
        kwargs = api.create_commit.call_args.kwargs
        assert kwargs["parent_commit"] == SHA
        assert [op.path_in_repo for op in kwargs["operations"]] == ["models.py"]
    else:
        assert result == SHA
        api.create_commit.assert_not_called()
    assert all(call.kwargs["revision"] == SHA for call in download.call_args_list)


def test_space_read_failure_never_commits(monkeypatch):
    api = Mock(token=False)
    monkeypatch.setattr(deployment, "check_space", Mock(return_value=SimpleNamespace(sha=SHA)))
    monkeypatch.setattr(deployment, "hf_hub_download", Mock(side_effect=OSError("offline")))
    with pytest.raises(OSError):
        deployment.deploy(api, "example/S1MB-leaderboard",
                          deployment.payload(IMAGE, SHA, ["example/model"]), SHA)
    api.create_commit.assert_not_called()


def test_card_model_metadata_matches_generated_list():
    import ast
    files = deployment.payload(IMAGE, SHA, ["z/model", "A/model", "z/model"])
    card = deployment.SpaceCard(files["README.md"].decode())
    names = ast.literal_eval(ast.parse(files["models.py"]).body[1].value)
    assert card.data.models == names == ["A/model", "z/model"]
    assert card.data.sdk == "docker"
    assert card.data.app_port == 7860
    assert "# S1MB Leaderboard" in card.text
    empty = deployment.payload(IMAGE, SHA, [])
    assert deployment.SpaceCard(empty["README.md"].decode()).data.models == []


@pytest.mark.parametrize("private", [True, False])
def test_visibility_changes_during_startup_are_rejected(monkeypatch, private):
    api = Mock()
    api.space_info.return_value = SimpleNamespace(private=not private, sha=SHA)
    with pytest.raises(RuntimeError, match="visibility changed"):
        deployment.wait_ready(api, "example/S1MB-leaderboard", SHA, 1, private=private)
    api.get_space_runtime.assert_not_called()


def test_model_references_use_bundled_measurement_revision(tmp_path, monkeypatch):
    import json
    file = tmp_path / "summary.json"
    file.write_text(json.dumps({"version": 1, "source": {"repo": "example/results", "revision": SHA}}))
    fetch = Mock(return_value=str(file))
    monkeypatch.setattr(deployment, "hf_hub_download", fetch)
    assert deployment.display_source_revision(Mock(token=False), "example/results", "c" * 40) == SHA
    assert fetch.call_args.kwargs["revision"] == "c" * 40
    file.write_text(json.dumps({"version": 1, "source": {"repo": "other/results", "revision": SHA}}))
    with pytest.raises(ValueError):
        deployment.display_source_revision(Mock(token=False), "example/results", "c" * 40)


def test_bounded_training_metadata_can_cover_both_languages(tmp_path, monkeypatch):
    exposure = {"benchmark_id": "synthetic", "relationship": "same_source_family", "dataset": "example/train", "revision": "a" * 40, "config": "synthetic", "split": "train", "evidence_url": "https://example.org/provenance", "note": "x" * 800}
    api, download = metadata_snapshot(tmp_path, [{"hf_url": "https://huggingface.co/example/model", "training_exposure": [exposure] * 164}])
    assert 65536 < api.dataset_info.return_value.siblings[0].size < 1024 * 1024
    monkeypatch.setattr(deployment, "hf_hub_download", download)
    assert deployment.published_models(api, "example/results") == ["example/model"]


@pytest.mark.parametrize("data,status", [
    ({"version": 1, "payload": {"results": [1]}}, 200),
    ({"error": "unavailable"}, 503),
    ({"version": 1, "payload": {"results": []}}, 200),
    ({"version": 1, "payload": {"results": "invalid"}}, 200),
    ("invalid JSON", 200),
])
def test_async_readiness_requires_successful_display_api(monkeypatch, data, status):
    import json
    from contextlib import contextmanager
    api = Mock(token=False)
    api.space_info.return_value = SimpleNamespace(private=False, sha=SHA,
        host="https://example-s1mb-leaderboard.hf.space")
    api.get_space_runtime.return_value = SimpleNamespace(stage="RUNNING", raw={"sha": SHA})
    requests = []

    @contextmanager
    def stream(method, url, **kwargs):
        requests.append(url)
        shell = url.endswith(".space/")
        body = b"Preparing the leaderboard" if shell else (
            data.encode() if isinstance(data, str) else json.dumps(data).encode())
        yield SimpleNamespace(status_code=200 if shell else status,
                              iter_bytes=lambda: iter([body]))

    monkeypatch.setattr(deployment.httpx, "get", Mock(return_value=SimpleNamespace(status_code=200, text="Preparing the leaderboard")))
    monkeypatch.setattr(deployment.httpx, "stream", stream)
    monkeypatch.setattr(deployment.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    monkeypatch.setattr(deployment.time, "sleep", Mock())
    if data == {"version": 1, "payload": {"results": [1]}}:
        deployment.wait_ready(api, "example/S1MB-leaderboard", SHA, 1, private=False)
    else:
        with pytest.raises(TimeoutError):
            deployment.wait_ready(api, "example/S1MB-leaderboard", SHA, 1, private=False)
    assert requests == ["https://example-s1mb-leaderboard.hf.space/",
                        "https://example-s1mb-leaderboard.hf.space/api/display/"]


def test_readiness_response_is_bounded_and_closes_the_stream(monkeypatch):
    from contextlib import contextmanager
    closed = []

    @contextmanager
    def stream(*args, **kwargs):
        try:
            yield SimpleNamespace(status_code=200, iter_bytes=lambda: iter([b"123", b"456"]))
        finally:
            closed.append(True)

    monkeypatch.setattr(deployment.httpx, "stream", stream)
    assert deployment.read_http_body("https://example.hf.space/", {}, 4) is None
    assert closed == [True]
