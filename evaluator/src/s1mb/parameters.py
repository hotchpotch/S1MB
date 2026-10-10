"""Count unique model parameters, excluding embedding weights from static AP."""

import argparse
import importlib
import json
import math
from types import FunctionType, ModuleType
from typing import Any

METHOD = "embedding_excluded_parameters_v1"
STATIC_EMBEDDINGS = {
    "position_embedding_table", "pos_embedding", "position_embedding",
    "position_embeddings", "pos_embed", "class_embedding", "cls_token", "mask_token",
}


def parameter_metadata(model: Any) -> dict[str, Any]:
    """Count complete modules without inference or device transfers.

    Include frozen parameters and heads; exclude buffers. Subtract all embedding
    weights even when shared with an output head. AP is a static size metric,
    not a routed MoE or per-input operation count.
    """
    torch = importlib.import_module("torch")
    modules: dict[int, Any] = {}
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.nn.Module):
            modules.update((id(module), module) for module in value.modules())
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif hasattr(value, "__dict__") and not isinstance(value, (type, ModuleType, FunctionType)):
            for child in vars(value).values():
                visit(child)

    visit(model)
    if not modules:
        raise ValueError("No torch modules found; cannot count model parameters")
    parameters: dict[int, Any] = {}
    lookup: set[int] = set()
    for module in modules.values():
        is_lookup = isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag))
        for name, parameter in module.named_parameters(recurse=False):
            parameters[id(parameter)] = parameter
            if (is_lookup and name == "weight") or name in STATIC_EMBEDDINGS:
                lookup.add(id(parameter))
    excluded = lookup

    def logical_size(parameter: Any) -> int:
        # Packed NF4 storage has fewer elements than the original weight matrix.
        quantization = getattr(parameter, "quant_state", None)
        shape = getattr(quantization, "shape", None)
        if shape is not None:
            if not shape or any(int(size) <= 0 for size in shape):
                raise ValueError("Invalid quantized parameter shape")
            return math.prod(int(size) for size in shape)
        return parameter.numel()

    return {
        "total_params": sum(logical_size(p) for p in parameters.values()),
        "active_params": sum(
            logical_size(p) for key, p in parameters.items() if key not in excluded
        ),
        "parameter_count_method": METHOD,
    }


def main() -> None:
    """Load a caller-supplied model factory and print reusable JSON metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factory", help="Python module:function returning the complete model")
    parser.add_argument("--kwargs", default="{}", help="JSON keyword arguments for the factory")
    args = parser.parse_args()
    module, separator, name = args.factory.partition(":")
    if not separator or not module or not name:
        parser.error("factory must be module:function")
    try:
        kwargs = json.loads(args.kwargs)
    except ValueError as exc:
        parser.error(str(exc))
    if not isinstance(kwargs, dict):
        parser.error("--kwargs must be a JSON object")
    factory: Any = importlib.import_module(module)
    for attribute in name.split("."):
        factory = getattr(factory, attribute)
    print(json.dumps(parameter_metadata(factory(**kwargs)), indent=2))


if __name__ == "__main__":
    main()
