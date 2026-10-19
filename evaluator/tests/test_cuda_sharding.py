"""Reject offload and retain activation space in native decoder placement."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.cuda_sharding import dispatch_decision_tower, two_gpu_cut


def test_contiguous_split_accounts_for_vision_and_embeddings():
    assert two_gpu_cut([10] * 6, 20, [60, 60]) == 2


def test_rejects_weights_that_only_fit_without_activation_reserve():
    with pytest.raises(ValueError, match="CPU offload is disabled"):
        two_gpu_cut([30, 30], 0, [33, 33])


def test_requires_two_devices():
    with pytest.raises(ValueError, match="Two CUDA"):
        two_gpu_cut([10, 10], 0, [100])


def test_rejects_unknown_native_decision_tower_before_dispatch():
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
    with pytest.raises(ValueError, match="decision tower structure"):
        dispatch_decision_tower(SimpleNamespace(), torch)


def test_decision_dispatch_never_preloads_all_layers_on_one_gpu(monkeypatch):
    """Model Accelerate's recursive placement, including direct calibration tensors."""
    output_hooks = []
    tower = SimpleNamespace(named_children=lambda: [
        (name, object()) for name in ["embed_tokens", "layers", "norm", "rotary_emb"]
    ], layers=[SimpleNamespace(register_forward_hook=output_hooks.append) for _ in range(6)])
    direct = {"mix_logits": "cpu", "cal_logT": "cpu"}
    model = SimpleNamespace(
        tower=tower,
        named_children=lambda: [("tower", tower), ("visual", object()), ("scorer", object())],
        named_parameters=lambda recurse: [("mix_logits", object())],
        named_buffers=lambda recurse: [("cal_logT", object())],
        parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="cuda"))],
        exit_indices=lambda: [3],
    )
    sizes = {"": 80, "tower": 70, "tower.layers": 60,
             **{f"tower.layers.{i}": 10 for i in range(6)}}
    moved = []

    def move(module, name, device):
        assert module is model
        direct[name] = device
        moved.append(name)

    def dispatch(module, device_map, main_device, force_hooks):
        assert module is model and main_device == 0 and force_hooks
        # Accelerate recursively places an explicitly mapped ancestor before
        # installing descendant hooks. Either ancestor exceeds this GPU's 60 bytes.
        for name in device_map:
            if sizes.get(name, 0) > 60:
                raise MemoryError("Ancestor placement loads the complete tower")
        assert set(direct.values()) == {"cuda:0"}

    accelerate = SimpleNamespace(
        utils=SimpleNamespace(compute_module_sizes=lambda module: sizes,
                              set_module_tensor_to_device=move,
                              send_to_device=lambda output, device: (output, device)),
        dispatch_model=dispatch,
    )
    monkeypatch.setattr("s1mb.adapters.cuda_sharding.importlib.import_module", lambda _: accelerate)
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2,
                                                 mem_get_info=lambda _: (60, 60)))
    placement = dispatch_decision_tower(model, torch)
    assert moved == ["mix_logits", "cal_logT"]
    assert placement["tower.layers.0"] == 0 and placement["tower.layers.5"] == 1
    assert placement["visual"] == placement["scorer"] == placement["cal_logT"] == 0
    assert len(output_hooks) == 2
    assert output_hooks[0](tower.layers[-1], (), "activations") == ("activations", "cuda:0")


@pytest.mark.parametrize("placement, rejected", [({"layers": 0}, False), ({"layers": "cpu"}, True), ({"layers": "disk"}, True)])
def test_meta_planning_never_accepts_cpu_or_disk(monkeypatch, placement, rejected):
    from contextlib import nullcontext

    from s1mb.adapters.cuda_sharding import cuda_only_device_map

    model = SimpleNamespace(tie_weights=lambda: None, _no_split_modules=["Decoder"])

    def infer(module, **kwargs):
        assert module is model
        assert kwargs["max_memory"] == {0: 60, 1: 60}
        assert kwargs["no_split_module_classes"] == ["Decoder"]
        return placement

    accelerate = SimpleNamespace(init_empty_weights=nullcontext, infer_auto_device_map=infer)
    monkeypatch.setattr("s1mb.adapters.cuda_sharding.importlib.import_module", lambda name: accelerate)
    torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(device_count=lambda: 2, mem_get_info=lambda i: (70, 70)))
    if rejected:
        with pytest.raises(ValueError, match="CPU/disk offload"):
            cuda_only_device_map(lambda config: model, object(), torch, activation_reserve=10)
    else:
        assert cuda_only_device_map(lambda config: model, object(), torch, activation_reserve=10) == placement


def test_tied_parameters_are_restored_after_meta_assignment_hooks(monkeypatch):
    from contextlib import contextmanager

    from s1mb.adapters.cuda_sharding import cuda_only_device_map

    state = {"initializing": False, "tied": False}

    @contextmanager
    def initialize():
        state["initializing"] = True
        yield
        state["initializing"] = False

    def tie():
        assert not state["initializing"]
        state["tied"] = True

    model = SimpleNamespace(tie_weights=tie, _no_split_modules=[])

    def construct(config):
        assert state["initializing"]
        return model

    def infer(module, **kwargs):
        assert state["tied"]
        return {"": 0}

    accelerate = SimpleNamespace(init_empty_weights=initialize, infer_auto_device_map=infer)
    monkeypatch.setattr("s1mb.adapters.cuda_sharding.importlib.import_module", lambda name: accelerate)
    torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(
        device_count=lambda: 2, mem_get_info=lambda i: (100, 100)))
    assert cuda_only_device_map(construct, object(), torch, activation_reserve=10) == {"": 0}


def test_explicit_activation_reserve_balances_tightly_fitting_layers():
    # A contiguous CUDA-only split can fit without a swap/offload reserve.
    assert two_gpu_cut([116] * 48, 280, [3070, 3180], activation_reserve=150) == 22
    with pytest.raises(ValueError, match="CPU offload"):
        two_gpu_cut([116] * 48, 280, [3070, 3180], activation_reserve=200)
    with pytest.raises(ValueError, match="positive"):
        two_gpu_cut([10, 10], 0, [100, 100], activation_reserve=0)
