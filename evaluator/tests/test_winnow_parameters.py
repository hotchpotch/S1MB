"""Winnow logical shapes and shared output parameter counting."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from s1mb.adapters.winnow import winnow_parameter_metadata
from s1mb.parameters import METHOD


def reader(monkeypatch, tensors, architecture="gemma4"):
    fake = SimpleNamespace(
        fields={"general.architecture": SimpleNamespace(contents=lambda: architecture)},
        tensors=[SimpleNamespace(name=name, shape=shape) for name, shape in tensors],
    )
    monkeypatch.setattr(
        "s1mb.adapters.winnow.importlib.import_module",
        lambda name: SimpleNamespace(GGUFReader=lambda path: fake),
    )


@pytest.mark.parametrize("separate_output", [False, True])
def test_logical_shapes_and_tied_output(monkeypatch, separate_output):
    tensors = [("token_embd.weight", [32, 64]), ("blk.0.attn_q.weight", [32, 32])]
    if separate_output:
        tensors.append(("output.weight", [32, 64]))
    reader(monkeypatch, tensors)
    assert winnow_parameter_metadata("unused.gguf") == {
        "total_params": 5120 if separate_output else 3072,
        "active_params": 3072 if separate_output else 1024,
        "parameter_count_method": "embedding_excluded_parameters_v1",
    }


@pytest.mark.parametrize(
    "tensors",
    [
        [],
        [("token_embd.weight", [0, 64])],
        [("token_embd.weight", [32, 64])] * 2,
        [("token_embd.weight", [32, 64]), ("v.patch_embd.weight", [32, 64])],
    ],
)
def test_reject_invalid_or_unsupported_tensors(monkeypatch, tensors):
    reader(monkeypatch, tensors)
    with pytest.raises(ValueError):
        winnow_parameter_metadata("unused.gguf")


def test_reject_other_architectures(monkeypatch):
    reader(monkeypatch, [("token_embd.weight", [32, 64])], architecture="other")
    with pytest.raises(ValueError, match="Gemma 4"):
        winnow_parameter_metadata("unused.gguf")


def test_per_layer_token_lookup_counts_total_but_not_active(monkeypatch):
    reader(monkeypatch, [("token_embd.weight", [32, 64]),
                        ("per_layer_token_embd.weight", [128, 64]),
                        ("blk.0.attn_q.weight", [32, 32])])
    assert winnow_parameter_metadata("unused.gguf") == {
        "total_params": 11264,
        "active_params": 1024,
        "parameter_count_method": METHOD,
    }


@pytest.mark.parametrize("capacity", [65, 77, 151, 255])
def test_native_capacity_cannot_be_overridden_above_64(capacity):
    from s1mb.adapters.winnow import WinnowAdapter

    with pytest.raises(ValueError, match="native candidate capacity.*2..64"):
        WinnowAdapter("unused", "unused", "unused", "cuda:0", max_candidates=capacity)


def test_vision_preserves_distinct_source_and_projector_fingerprints(monkeypatch, tmp_path):
    from s1mb.adapters import winnow

    source_digest = hashlib.sha256(b"python source").hexdigest()
    weights = tmp_path / "weights.gguf"
    projector = tmp_path / "projector.gguf"
    weights.write_bytes(b"weights")
    projector.write_bytes(b"projector")
    artifact = {"file": weights.name, "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "bytes": weights.stat().st_size}
    projector_digest = hashlib.sha256(projector.read_bytes()).hexdigest()
    projection = {"url": "https://example.org/resolve/main/projector.gguf",
                  "sha256": projector_digest, "bytes": projector.stat().st_size}
    (tmp_path / "manifests").mkdir()
    (tmp_path / ".build/bin").mkdir(parents=True)
    (tmp_path / ".build/bin/winnow-server").write_bytes(b"server")
    (tmp_path / "manifests/models.json").write_text(json.dumps({"release": {"model": artifact}}))
    (tmp_path / "manifests/release-assets-v1.json").write_text(
        json.dumps({"models": {"12b-q8": {"projector": projection}}}))
    (tmp_path / "runtime.lock.json").write_text("{}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(winnow, "source_path", lambda source: (tmp_path, source_digest))
    monkeypatch.setattr(winnow, "winnow_parameter_metadata",
                        lambda path: {"total_params": 100, "active_params": 80,
                                      "parameter_count_method": METHOD})
    hub = SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40),
                          hf_hub_download=lambda model, file, **kwargs: str(tmp_path / file))
    gguf = SimpleNamespace(GGUFReader=lambda path: SimpleNamespace(fields={}, tensors=[]))
    monkeypatch.setattr(winnow.importlib, "import_module",
                        lambda name: {"huggingface_hub": hub, "gguf": gguf}[name])

    class BeforeServerStartup(Exception):
        pass

    def stop_before_server(*args):
        raise BeforeServerStartup

    monkeypatch.setattr(winnow, "check_port_available", stop_before_server)
    adapter = winnow.WinnowAdapter.__new__(winnow.WinnowAdapter)
    with pytest.raises(BeforeServerStartup):
        adapter.__init__("EldanRing/Winnow-12B", "a" * 40, str(tmp_path), "cuda:0", vision=True)
    assert adapter.info.settings["source_python_sha256"] == source_digest
    assert adapter.info.settings["projector_sha256"] == projector_digest
    assert source_digest != projector_digest
