"""Resolve display parameter counts from public checkpoint headers, without weights."""

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.errors import HfHubHTTPError

from .data import ModelInfo, read_json, write_json
from .gguf_headers import gguf_shapes
from .parameters import METHOD
from .result_repository import ModelMetadata


class UnsupportedCheckpoint(ValueError):
    """Headers do not establish the project's logical parameter counting rules."""


def needs_static_recount(total: int | None, active: int | None, method: str | None) -> bool:
    """Select legacy equal/near-equal counts that may retain tied token embeddings."""
    return (total is not None and active is not None and method != METHOD
            and 0 <= total - active <= total // 100)


FULL_VISION_ADAPTERS = {"rune", "jev-omni", "tev", "reflex", "metask", "apus", "imajev"}


def checkpoint_counts(config: dict[str, Any], metadata: Any, *, measured_total: int | None = None,
                      text_only: bool = False) -> dict[str, Any]:
    """Count static embedding-excluded AP for supported Qwen and Gemma checkpoints.

    A supplied measured total preserves loaded heads and excludes saved buffers.
    This is static AP, not a routed/executed parameter count. Quantized
    packed tensors and unrecognized architectures require measured counts.
    """
    text = config.get("text_config", config)
    supported = {"qwen2", "qwen3", "qwen3_5", "qwen3_5_text", "gemma4",
                 "gemma4_text", "gemma4_unified", "gemma4_unified_text"}
    if (config.get("model_type") not in supported or text.get("model_type") not in supported
            or config.get("quantization_config") or text.get("quantization_config")):
        raise UnsupportedCheckpoint("Unsupported architecture or packed quantization")
    sizes: dict[str, int] = {}
    names: set[str] = set()
    for file in metadata.files_metadata.values():
        for name, tensor in file.tensors.items():
            if name in names:
                raise UnsupportedCheckpoint("Duplicate tensor names")
            names.add(name)
            if name.endswith((".layer_scalar", ".std_bias", ".std_scale")):
                continue  # Registered Gemma buffers, not model parameters.
            if (
                tensor.dtype not in {"F64", "F32", "F16", "BF16"}
                or not name.endswith((".weight", ".bias", ".A_log", ".dt_bias",
                                     ".layer_scalar", ".scale", ".per_expert_scale",
                                     ".down_proj", ".gate_up_proj", ".pos_embedding",
                                     ".position_embedding_table"))
                or any(type(d) is not int or d < 1 for d in tensor.shape)
            ):
                raise UnsupportedCheckpoint(f"Unsupported parameter tensor: {name}")
            sizes[name] = math.prod(tensor.shape)
    if names != set(metadata.weight_map):
        raise UnsupportedCheckpoint("Checkpoint index/header inventory mismatch")
    embeddings = [n for n in sizes if n in {
        "model.embed_tokens.weight", "model.language_model.embed_tokens.weight",
    }]
    if len(embeddings) != 1:
        raise UnsupportedCheckpoint("Expected one text embedding table")
    embedding = embeddings[0]
    shape = text.get("vocab_size"), text.get("hidden_size")
    if any(type(d) is not int or d < 1 for d in shape) or math.prod(shape) != sizes[embedding]:
        raise UnsupportedCheckpoint("Text embedding shape disagrees with config")
    tied = text.get("tie_word_embeddings", config.get("tie_word_embeddings"))
    if type(tied) is not bool:
        raise UnsupportedCheckpoint("Missing explicit embedding sharing configuration")
    if not tied and "lm_head.weight" not in sizes:
        raise UnsupportedCheckpoint("Missing untied output head")
    # A saved tied output alias describes the same parameter, not a second matrix.
    if tied and "lm_head.weight" in sizes:
        if sizes["lm_head.weight"] != sizes[embedding]:
            raise UnsupportedCheckpoint("Tied output shape disagrees with embeddings")
        del sizes["lm_head.weight"]
    total = sum(sizes.values()) if measured_total is None else measured_total
    excluded = sizes[embedding]
    # Qwen 3.5's vision position table is nn.Embedding; patch projection is active.
    excluded += sizes.get("model.language_model.embed_tokens_per_layer.weight", 0)
    if not text_only:
        excluded += sum(sizes.get(name, 0) for name in (
            "model.visual.pos_embed.weight", "model.vision_embedder.pos_embedding",
            "model.vision_tower.patch_embedder.position_embedding_table",
        ))
    if excluded > total:
        raise UnsupportedCheckpoint("Embedding parameters exceed measured total")
    return {"total_params": total, "active_params": total - excluded,
            "parameter_count_method": METHOD}


def resolve_counts(repo_id: str, revision: str = "main", *, api: Any = None,
                   measured_total: int | None = None, subfolder: str = "",
                   text_only: bool = False) -> dict[str, Any]:
    """Pin the revision before reading config and all Safetensors headers."""
    api = api if api is not None else HfApi(token=False)
    info = api.model_info(repo_id, revision=revision)
    if not info.sha or not re.fullmatch(r"[a-f0-9]{40}", info.sha):
        raise ValueError("Hub did not resolve an immutable model revision")
    if subfolder and not re.fullmatch(r"[\w-]+(?:/[\w-]+)*", subfolder):
        raise UnsupportedCheckpoint("Invalid checkpoint subfolder")
    prefix = subfolder + "/" if subfolder else ""
    config_path = Path(hf_hub_download(repo_id, prefix + "config.json",
                                     revision=info.sha, token=False))
    if config_path.stat().st_size > 1024 * 1024:
        raise UnsupportedCheckpoint("Oversized model config")
    config = read_json(config_path)
    if subfolder:
        files = [f.rfilename for f in info.siblings or [] if f.rfilename.startswith(prefix)
                 and "/" not in f.rfilename[len(prefix):] and f.rfilename.endswith(".safetensors")]
        if not files or len(files) > 100:
            raise UnsupportedCheckpoint("Invalid subfolder Safetensors inventory")
        parsed = {f: api.parse_safetensors_file_metadata(repo_id, f, revision=info.sha,
                                                        token=False) for f in files}
        metadata = SimpleNamespace(files_metadata=parsed,
                                   weight_map={n: f for f, p in parsed.items() for n in p.tensors})
    else:
        metadata = api.get_safetensors_metadata(repo_id, revision=info.sha, token=False)
    return {**checkpoint_counts(config, metadata, measured_total=measured_total, text_only=text_only),
            "repo_id": repo_id, "revision": info.sha, "subfolder": subfolder or None}


def resolve_model_counts(model: ModelInfo, repo: str, revision: str, total: int,
                         *, api: Any = None) -> dict[str, Any]:
    """Recompute static AP against measured TP, preserving extra loaded heads."""
    api = api if api is not None else HfApi(token=False)
    if model.adapter == "winnow":
        info = api.model_info(repo, revision=revision, files_metadata=True)
        digest = model.settings.get("weights_sha256")
        files = [s.rfilename for s in info.siblings or [] if s.rfilename.endswith(".gguf")
                 and s.lfs is not None and s.lfs.sha256 == digest]
        if len(files) != 1:
            raise UnsupportedCheckpoint("Cannot identify recorded GGUF by weights SHA256")
        shapes = gguf_shapes(hf_hub_url(repo, files[0], revision=info.sha))
        if "token_embd.weight" not in shapes or sum(math.prod(s) for s in shapes.values()) != total:
            raise UnsupportedCheckpoint("GGUF shapes disagree with measured total")
        excluded = sum(math.prod(s) for n, s in shapes.items() if n in {
            "token_embd.weight", "per_layer_token_embd.weight", "position_embd.weight",
        })
        return {"total_params": total, "active_params": total - excluded,
                "parameter_count_method": METHOD, "repo_id": repo, "revision": info.sha,
                "filename": files[0]}
    base, base_revision = model.settings.get("base_model"), model.settings.get("base_revision")
    if base is not None:
        if not isinstance(base, str) or not isinstance(base_revision, str) or not re.fullmatch(
            r"[a-f0-9]{40}", base_revision
        ):
            raise UnsupportedCheckpoint("Base checkpoint requires an exact recorded revision")
        result = resolve_counts(base, base_revision, api=api, measured_total=total,
                                text_only=model.adapter not in FULL_VISION_ADAPTERS)
        if model.adapter == "bosun":
            # Bosun resizes the input table to include authored decision tokens.
            config = read_json(Path(hf_hub_download(repo, "config.json", revision=revision,
                                                   token=False)))
            decision = api.parse_safetensors_file_metadata(
                repo, config["decision_embeddings_file"], revision=revision, token=False)
            inputs, outputs = (decision.tensors.get(n) for n in (
                "input_embeddings", "output_embeddings"))
            if (inputs is None or outputs is None or inputs.shape != outputs.shape
                    or len(inputs.shape) != 2 or inputs.shape[0] != config["decision_token_count"]):
                raise UnsupportedCheckpoint("Cannot verify Bosun added embedding rows")
            result["active_params"] = total - config["vocab_size"] * inputs.shape[1]
        return result | {"source_model_repo": repo, "source_model_revision": revision}
    subfolder = "general/backbone" if model.adapter == "minojev" else ""
    return resolve_counts(repo, revision, api=api, measured_total=total, subfolder=subfolder,
                          text_only=model.adapter not in FULL_VISION_ADAPTERS)


def hub_id(url: str | None) -> str | None:
    """Only explicit root checkpoint links identify a model to inspect."""
    if not url:
        return None
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or len(parts) != 2:
        return None
    if parts[0] in {"datasets", "spaces"}:
        return None
    return "/".join(parts) if all(re.fullmatch(r"[\w.-]+", p) for p in parts) else None


def enrich_metadata(repository: Path, *, api: Any = None) -> list[dict[str, Any]]:
    """Fill missing counts and migrate legacy near-equal AP/TP in a publication copy only.

    Preserve measured totals and current-method counts; recalculate legacy gaps <=1%.
    Alias revisions use the current linked checkpoint for display only; the report
    records this distinction and never claims to recover an evaluation revision.
    """
    report = []
    for path in sorted(repository.glob("*/metadata.json")):
        display = ModelMetadata.model_validate(read_json(path))
        repo = hub_id(display.hf_url or display.url)
        if repo is None:
            continue
        if (display.total_params is not None and display.active_params is not None
                and not needs_static_recount(display.total_params, display.active_params,
                                             display.parameter_count_method)):
            continue
        models = [ModelInfo.model_validate(read_json(p)["model"])
                  for p in sorted(path.parent.glob("*.json.xz"))]
        if not models:
            continue
        existing = {(m.total_params, m.active_params, m.parameter_count_method) for m in models}
        if display.total_params is None and display.active_params is None:
            if len(existing) != 1:
                report.append({"model_id": display.model_id, "status": "unresolved",
                               "reason": "Conflicting measured parameter counts"})
                continue
            total, active, method = next(iter(existing))
        else:
            total, active = display.total_params, display.active_params
            method = display.parameter_count_method
        recalculate = needs_static_recount(total, active, method)
        if total is not None and active is not None and not recalculate:
            continue
        revisions = {m.revision for m in models}
        exact = all(m.id == repo for m in models) and all(
            re.fullmatch(r"[a-f0-9]{40}", r) for r in revisions)
        if exact and len(revisions) != 1:
            report.append({"model_id": display.model_id, "status": "unresolved",
                           "reason": "Multiple evaluated checkpoint revisions"})
            continue
        revision = next(iter(revisions)) if exact else "main"
        try:
            if recalculate and total is not None:
                count_settings = {json.dumps({k: m.settings.get(k) for k in (
                    "base_model", "base_revision", "weights_sha256", "checkpoint_subdir"
                )}, sort_keys=True) for m in models}
                if len(count_settings) != 1:
                    raise UnsupportedCheckpoint("Conflicting checkpoint counting settings")
                counts = resolve_model_counts(models[0], repo, revision, total, api=api)
            else:
                counts = resolve_counts(repo, revision, api=api)
            if (total is not None and total != counts["total_params"]) or (
                active is not None and not recalculate and active != counts["active_params"]
            ):
                raise UnsupportedCheckpoint("Saved parameter counts disagree with checkpoint")
        except (UnsupportedCheckpoint, HfHubHTTPError) as exc:
            report.append({"model_id": display.model_id, "status": "unresolved",
                           "reason": str(exc)})
            continue
        enriched = display.model_dump() | {k: counts[k] for k in (
            "total_params", "active_params", "parameter_count_method")}
        ModelMetadata.model_validate(enriched)
        write_json(path, enriched)
        report.append({"model_id": display.model_id, "status": "resolved", **counts,
                       "previous_active_params": active,
                       "source": "evaluated-checkpoint" if exact else "linked-checkpoint"})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    print(json.dumps(resolve_counts(args.repo_id, args.revision), indent=2))


if __name__ == "__main__":
    main()
