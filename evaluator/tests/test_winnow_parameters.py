"""Winnow logical shapes and shared output parameter counting."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.winnow import winnow_parameter_metadata


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
        "active_params": 3072,
        "parameter_count_method": "non_lookup_parameters_v1",
    }


@pytest.mark.parametrize(
    "tensors",
    [
        [],
        [("token_embd.weight", [0, 64])],
        [("token_embd.weight", [32, 64])] * 2,
        [("token_embd.weight", [32, 64]), ("per_layer_token_embd.weight", [32, 64])],
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
