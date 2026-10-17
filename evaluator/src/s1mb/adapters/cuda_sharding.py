"""Explicit CUDA-only placement for complete native Qwen vision decoder layers."""

import importlib


def two_gpu_cut(layer_bytes, fixed_bytes, free_bytes):
    """Reserve activation space and balance a contiguous split across two GPUs."""
    if len(free_bytes) != 2 or len(layer_bytes) < 2:
        raise ValueError("Two CUDA devices and at least two layers are required")
    budgets = [int(free * 0.88) for free in free_bytes]
    candidates = []
    for cut in range(1, len(layer_bytes)):
        used = [fixed_bytes + sum(layer_bytes[:cut]), sum(layer_bytes[cut:])]
        if all(size <= budget for size, budget in zip(used, budgets, strict=True)):
            candidates.append((max(size / budget for size, budget in zip(used, budgets, strict=True)), cut))
    if not candidates:
        raise ValueError("Native weights exceed reserved CUDA memory; CPU offload is disabled")
    return min(candidates)[1]


def dispatch_vision_decoder(backbone, torch):
    """Move a CPU-loaded backbone to two CUDA devices before any inference."""
    if torch.cuda.device_count() != 2:
        raise ValueError("CUDA sharding requires exactly two visible GPUs")
    if set(dict(backbone.named_children())) != {"visual", "language_model"} or set(
        dict(backbone.language_model.named_children())
    ) != {"embed_tokens", "layers", "norm", "rotary_emb"}:
        raise ValueError("Unsupported native vision decoder structure")
    accelerate = importlib.import_module("accelerate")
    sizes = accelerate.utils.compute_module_sizes(backbone)
    names = [f"language_model.layers.{i}" for i in range(len(backbone.language_model.layers))]
    layers = [sizes[name] for name in names]
    free = [torch.cuda.mem_get_info(i)[0] for i in range(2)]
    cut = two_gpu_cut(layers, sizes[""] - sum(layers), free)
    placement = {
        "visual": 0, "language_model.embed_tokens": 0,
        "language_model.norm": 0, "language_model.rotary_emb": 0,
        **{name: int(i >= cut) for i, name in enumerate(names)},
    }
    accelerate.dispatch_model(backbone, device_map=placement, main_device=0, force_hooks=True)
    if any(parameter.device.type != "cuda" for parameter in backbone.parameters()):
        raise RuntimeError("Native backbone contains non-CUDA weights after dispatch")
    return placement
