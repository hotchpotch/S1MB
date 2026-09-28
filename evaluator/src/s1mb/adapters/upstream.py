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


def checkpoint_path(model: str, revision: str) -> tuple[Path, str]:
    if Path(model).exists():
        root = Path(model).resolve()
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
        hub.snapshot_download(model, revision=resolved, ignore_patterns=["*.gguf", "*.onnx"])
    )
    return root, resolved


class UpstreamAdapter:
    attention_model: Any

    def setup(self, name: str, model: str, revision: str, source: str, device: str):
        if not device.startswith("cuda"):
            raise ValueError("These evaluation bridges require explicit CUDA; no CPU fallback")
        self.name, self.model_id, self.device = name, model, device
        self.source, self.source_digest = source_path(source)
        self.path, self.revision = checkpoint_path(model, revision)
        self.torch = importlib.import_module("torch")
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
        self.torch.cuda.set_device(device)
        self.settings = {}

    def metadata(self) -> ModelInfo:
        return ModelInfo(
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


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


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
