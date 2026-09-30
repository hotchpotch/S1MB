"""Utilities for explicit, isolated upstream checkouts and pinned model snapshots."""

import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

from s1mb.data import ModelInfo
from s1mb.parameters import parameter_metadata


def source_path(source: str) -> tuple[Path, str]:
    root = Path(source).resolve(strict=True)
    for directory in [root, root / "src"]:
        if directory.is_dir():
            sys.path.insert(0, str(directory))
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if not any(part.startswith(".") for part in path.relative_to(root).parts):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return root, digest.hexdigest()


def checkpoint_path(model: str, revision: str, subfolder: str | None = None) -> tuple[Path, str]:
    if subfolder is not None and (
        Path(subfolder).name != subfolder or subfolder in {"", ".", ".."}
    ):
        raise ValueError("Checkpoint subfolder must be a single directory name")
    if Path(model).exists():
        root = (Path(model) / subfolder if subfolder else Path(model)).resolve(strict=True)
        digest = hashlib.sha256()
        for path in [root] if root.is_file() else sorted(root.rglob("*")):
            if path.is_file() and not any(p.startswith(".") for p in path.relative_to(root).parts):
                digest.update(str(path.relative_to(root)).encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        return root, digest.hexdigest()
    hub = importlib.import_module("huggingface_hub")
    resolved = hub.model_info(model, revision=revision).sha
    if not resolved:
        raise ValueError("Hub did not return a resolved checkpoint revision")
    root = Path(
        hub.snapshot_download(
            model,
            revision=resolved,
            ignore_patterns=["*.gguf", "*.onnx"],
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
        )
    )
    return root / subfolder if subfolder else root, resolved


class UpstreamAdapter:
    attention_model: Any
    engine: Any
    native: Any
    tokenizer: Any
    settings: dict[str, Any]

    def setup(self, name: str, model: str, revision: str, source: str, device: str, subfolder=None):
        if not device.startswith("cuda"):
            raise ValueError("These evaluation bridges require explicit CUDA; no CPU fallback")
        self.name, self.model_id, self.device = name, model, device
        self.source, self.source_digest = source_path(source)
        self.path, self.revision = checkpoint_path(model, revision, subfolder)
        self.torch = importlib.import_module("torch")
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
        self.torch.cuda.set_device(device)
        self.settings = {}

    def metadata(self) -> ModelInfo:
        model = self.engine._get_model() if self.name == "von" else self.engine
        return ModelInfo(
            **parameter_metadata(model),
            id=self.model_id,
            adapter=self.name,
            revision=self.revision,
            settings={
                "device": self.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "source_python_sha256": self.source_digest,
                "renderer": "upstream-native-dataset-default-instructions-v1",
                "questions_per_call": 1,
                "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers"]},
                **self.settings,
            },
        )

    def close(self):
        if hasattr(self, "attention_model"):
            del self.attention_model
        if hasattr(self, "engine"):
            del self.engine
        gc.collect()
        self.torch.cuda.empty_cache()

    def set_attention(self, implementation):
        """Select an explicit backend without removing model-specific attention masks."""
        if self.name == "von" and implementation != "sdpa":
            raise ValueError("Von's independent-option 4D masks require SDPA; FA2 is incompatible")
        actual = implementation
        if (
            implementation == "flash_attention_2"
            and int(importlib.metadata.version("transformers").split(".")[0]) >= 5
        ):
            actual = "kernels-community/flash-attn2@v3"
        self.attention_model.set_attn_implementation(actual)
        if self.attention_model.config._attn_implementation != actual:
            raise RuntimeError(f"Model did not activate requested attention backend: {actual}")
        self.settings["attention_implementation"] = actual

    def enable_kernels(self):
        """Use Transformers' supported kernels for hybrid recurrent layers."""
        model = self.attention_model
        if hasattr(model, "get_base_model"):
            model = model.get_base_model()
        kernels = importlib.import_module("kernels")
        hub = importlib.import_module("transformers.integrations.hub_kernels")
        # The Hub convolution layer builds require Torch >=2.11. Keep the native
        # GPU convolution while accelerating the expensive delta-rule recurrence.
        mapping = {
            **hub.get_kernel_mapping_transformers(),
            "causal_conv1d_fn": {},
            "causal_conv1d_update": {},
        }
        with kernels.use_kernel_mapping(mapping):
            kernels.kernelize(model, device="cuda", mode=kernels.Mode.INFERENCE)
        self.settings["hub_kernels"] = True
        self.settings["convolution_backend"] = "torch-gpu"
        self.settings["kernels_version"] = importlib.metadata.version("kernels")


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def decoder_device_map(model_path, device, torch, transformers):
    """Place complete decoder layers contiguously without offloading the output head."""
    count = torch.cuda.device_count()
    if count == 1:
        return {"": device}
    accelerate = importlib.import_module("accelerate")
    config = transformers.AutoConfig.from_pretrained(str(model_path))
    with accelerate.init_empty_weights():
        template = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    if set(dict(template.named_children())) != {"model", "lm_head"} or set(
        dict(template.model.named_children())
    ) != {"embed_tokens", "layers", "norm", "rotary_emb"}:
        raise ValueError("Unsupported decoder structure for explicit GPU placement")
    layers = len(template.model.layers)
    if layers < count:
        raise ValueError("There must be at least one decoder layer per visible GPU")
    placement = {
        "model.embed_tokens": 0,
        **{f"model.layers.{i}": i * count // layers for i in range(layers)},
        "model.norm": count - 1,
        "model.rotary_emb": count - 1,
        "lm_head": count - 1,
    }
    sizes = accelerate.utils.compute_module_sizes(template, dtype=torch.bfloat16)
    for gpu in range(count):
        required = sum(sizes[name] for name, target in placement.items() if target == gpu)
        if required > int(torch.cuda.mem_get_info(gpu)[0] * 0.88):
            raise ValueError(f"Decoder weights exceed the reserved GPU {gpu} memory budget")
    return placement


def candidate_batches(items, lengths, max_batch=8, token_budget=4096):
    """Bound padded candidate tokens; an oversized single path stays intact."""
    batch, width = [], 0
    for item, length in zip(items, lengths, strict=True):
        if batch and (
            len(batch) == max_batch or max(width, length) * (len(batch) + 1) > token_budget
        ):
            yield batch
            batch, width = [], 0
        batch.append(item)
        width = max(width, length)
    if batch:
        yield batch
