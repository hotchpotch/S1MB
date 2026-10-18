"""Kernel replacement preserves CUDA transfer wrappers without GPU dependencies."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from s1mb.adapters.upstream import UpstreamAdapter


@pytest.mark.parametrize("fail", [False, True])
def test_kernel_replacement_restores_transfer_wrapper(monkeypatch, fail):
    hook = object()
    layer = SimpleNamespace(_hf_hook=hook, forward=lambda: "old-wrapped")
    events = []

    def remove(module):
        events.append("remove")
        del module._hf_hook
        module.forward = lambda: "native"

    def kernelize(model, **kwargs):
        assert not hasattr(layer, "_hf_hook")
        assert kwargs["device"] == "cuda"
        layer.forward = lambda: "kernel"
        events.append("kernelize")
        if fail:
            raise RuntimeError("kernel setup failed")

    def add(module, original_hook):
        assert original_hook is hook
        events.append("add")
        kernel_forward = module.forward
        module.forward = lambda: ("transferred", kernel_forward())
        module._hf_hook = original_hook

    monkeypatch.setitem(sys.modules, "accelerate.hooks", SimpleNamespace(
        remove_hook_from_module=remove, add_hook_to_module=add,
    ))
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(
        use_kernel_mapping=lambda *args, **kwargs: nullcontext(),
        kernelize=kernelize, Mode=SimpleNamespace(INFERENCE="inference"),
    ))
    monkeypatch.setitem(sys.modules, "transformers.integrations.hub_kernels", SimpleNamespace(
        get_kernel_mapping_transformers=dict,
    ))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test")
    adapter = UpstreamAdapter.__new__(UpstreamAdapter)
    adapter.attention_model = SimpleNamespace(modules=lambda: [layer])
    adapter.settings = {}
    if fail:
        with pytest.raises(RuntimeError, match="kernel setup failed"):
            adapter.enable_kernels()
    else:
        adapter.enable_kernels()
    assert events == ["remove", "kernelize", "add"]
    assert layer.forward() == ("transferred", "kernel")
    assert layer._hf_hook is hook
