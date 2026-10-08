"""Checkpoint headers and publication enrichment use no network or model runtime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from s1mb import hub_parameters as hp
from s1mb.data import ModelInfo, read_json, write_json


def config(tied=True):
    return {"model_type": "qwen3_5_text", "vocab_size": 10, "hidden_size": 4,
            "tie_word_embeddings": tied}


def headers(**shapes):
    tensors = {n: SimpleNamespace(shape=s, dtype="BF16") for n, s in shapes.items()}
    return SimpleNamespace(files_metadata={"model.safetensors": SimpleNamespace(tensors=tensors)},
                           weight_map={n: "model.safetensors" for n in tensors})


def test_tied_output_is_counted_once_and_embedding_is_excluded():
    metadata = headers(**{"model.embed_tokens.weight": [10, 4],
                          "lm_head.weight": [10, 4], "model.layers.0.mlp.weight": [4, 4]})
    assert hp.checkpoint_counts(config(), metadata) == {
        "total_params": 56, "active_params": 16, "parameter_count_method": hp.METHOD}
    counts = hp.checkpoint_counts(config(False), metadata)
    assert counts["total_params"] == 96 and counts["active_params"] == 56


def test_vision_position_lookup_excluded_but_patch_projection_retained():
    metadata = headers(**{"model.language_model.embed_tokens.weight": [10, 4],
                          "model.visual.pos_embed.weight": [3, 4],
                          "model.visual.patch_embed.proj.weight": [4, 2]})
    counts = hp.checkpoint_counts({"model_type": "qwen3_5", "text_config": config()}, metadata)
    assert counts["total_params"] == 60 and counts["active_params"] == 8


@pytest.mark.parametrize("change", [
    {"model_type": "unknown"}, {"quantization_config": {"bits": 4}},
    {"tie_word_embeddings": None}, {"vocab_size": 11}, {"tie_word_embeddings": False},
])
def test_unsupported_or_incomplete_config_does_not_estimate(change):
    with pytest.raises(hp.UnsupportedCheckpoint):
        hp.checkpoint_counts(config() | change,
                             headers(**{"model.embed_tokens.weight": [10, 4]}))


def test_index_and_dtype_validation():
    metadata = headers(**{"model.embed_tokens.weight": [10, 4]})
    metadata.weight_map["missing.weight"] = "missing.safetensors"
    with pytest.raises(hp.UnsupportedCheckpoint, match="inventory"):
        hp.checkpoint_counts(config(), metadata)


def test_measured_total_preserves_heads_and_text_scope_excludes_unloaded_vision():
    metadata = headers(**{"model.language_model.embed_tokens.weight": [10, 4],
                          "model.visual.pos_embed.weight": [3, 4],
                          "model.visual.patch_embed.proj.weight": [4, 2]})
    counts = hp.checkpoint_counts({"model_type": "qwen3_5", "text_config": config()}, metadata,
                                  measured_total=56, text_only=True)
    assert counts["total_params"] == 56 and counts["active_params"] == 16


@pytest.mark.parametrize("adapter", ["tev", "reflex", "metask", "apus", "imajev"])
@pytest.mark.parametrize("has_base", [False, True])
def test_full_vision_runtime_recount_includes_position_embeddings(monkeypatch, adapter, has_base):
    resolve = Mock(return_value={"total_params": 1000, "active_params": 800})
    monkeypatch.setattr(hp, "resolve_counts", resolve)
    settings = {"base_model": "owner/base", "base_revision": "b" * 40} if has_base else {}
    model = ModelInfo(id="owner/model", adapter=adapter, revision="a" * 40, settings=settings)
    hp.resolve_model_counts(model, "owner/model", "a" * 40, 1000)
    assert resolve.call_args.kwargs["text_only"] is False
    assert resolve.call_args.kwargs["measured_total"] == 1000


def test_gemma_buffers_are_excluded_and_static_position_tables_are_subtracted():
    c = config() | {"model_type": "gemma4_text"}
    metadata = headers(**{"model.embed_tokens.weight": [10, 4],
                          "model.layers.0.weight": [4, 4],
                          "model.layers.0.layer_scalar": [1],
                          "model.vision_tower.std_bias": [4],
                          "model.vision_tower.patch_embedder.position_embedding_table": [2, 3, 4]})
    counts = hp.checkpoint_counts(c, metadata)
    assert counts["total_params"] == 80 and counts["active_params"] == 16
    metadata = headers(**{"model.embed_tokens.weight": [10, 4]})
    metadata.files_metadata["model.safetensors"].tensors["model.embed_tokens.weight"].dtype = "U8"
    with pytest.raises(hp.UnsupportedCheckpoint, match="tensor"):
        hp.checkpoint_counts(config(), metadata)


def test_resolves_sha_before_reads(monkeypatch, tmp_path):
    path = tmp_path / "config.json"
    write_json(path, config())
    download = Mock(return_value=str(path))
    monkeypatch.setattr(hp, "hf_hub_download", download)
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha="a" * 40)
    api.get_safetensors_metadata.return_value = headers(**{"model.embed_tokens.weight": [10, 4]})
    result = hp.resolve_counts("owner/model", "tag", api=api)
    assert result["revision"] == "a" * 40
    assert download.call_args.kwargs == {"revision": "a" * 40, "token": False}
    assert api.get_safetensors_metadata.call_args.kwargs["revision"] == "a" * 40


def staging(tmp_path, *, total=None, active=None, revision="a" * 40, model_id="owner/model"):
    folder = tmp_path / "owner__model"
    folder.mkdir()
    write_json(folder / "metadata.json", {
        "model_id": folder.name, "display_name": "Model", "short_name": "Model",
        "hf_url": "https://huggingface.co/owner/model", "total_params": total,
        "active_params": active, "parameter_count_method": hp.METHOD if active else None})
    write_json(folder / "test.json.xz", {"model": {
        "id": model_id, "adapter": "test", "revision": revision}})
    return folder


def resolver(monkeypatch):
    resolve = Mock(return_value={"total_params": 56, "active_params": 56,
                                "parameter_count_method": hp.METHOD,
                                "repo_id": "owner/model", "revision": "a" * 40})
    monkeypatch.setattr(hp, "resolve_counts", resolve)
    return resolve


def test_enrichment_preserves_measurements_and_fills_partial_metadata(monkeypatch, tmp_path):
    folder = staging(tmp_path, total=56)
    original = (folder / "test.json.xz").read_bytes()
    resolve = resolver(monkeypatch)
    report = hp.enrich_metadata(tmp_path)
    assert read_json(folder / "metadata.json")["active_params"] == 56
    assert (folder / "test.json.xz").read_bytes() == original
    assert report[0]["source"] == "evaluated-checkpoint"
    resolve.assert_called_once_with("owner/model", "a" * 40, api=None)


def test_saved_counts_win_and_disagreement_stays_unknown(monkeypatch, tmp_path):
    folder = staging(tmp_path, total=99)
    resolver(monkeypatch)
    original = (folder / "metadata.json").read_bytes()
    report = hp.enrich_metadata(tmp_path)
    assert report[0]["status"] == "unresolved"
    assert (folder / "metadata.json").read_bytes() == original


def test_complete_metadata_never_fetches(monkeypatch, tmp_path):
    staging(tmp_path, total=99, active=99)
    resolve = resolver(monkeypatch)
    assert hp.enrich_metadata(tmp_path) == []
    resolve.assert_not_called()


def test_api_only_rows_and_unsupported_headers_are_reported(monkeypatch, tmp_path):
    folder = staging(tmp_path)
    resolve = resolver(monkeypatch)
    resolve.side_effect = hp.UnsupportedCheckpoint("Unsupported architecture")
    original = (folder / "metadata.json").read_bytes()
    assert hp.enrich_metadata(tmp_path)[0]["reason"] == "Unsupported architecture"
    assert (folder / "metadata.json").read_bytes() == original
    m = read_json(folder / "metadata.json")
    write_json(folder / "metadata.json", m | {"hf_url": None, "url": "https://api.example.org"})
    resolve.reset_mock()
    assert hp.enrich_metadata(tmp_path) == []
    resolve.assert_not_called()


def test_complete_measured_counts_are_preserved(monkeypatch, tmp_path):
    folder = staging(tmp_path)
    result = read_json(folder / "test.json.xz")
    result["model"].update(total_params=99, active_params=80, parameter_count_method=hp.METHOD)
    write_json(folder / "test.json.xz", result)
    resolve = resolver(monkeypatch)
    assert hp.enrich_metadata(tmp_path) == []
    resolve.assert_not_called()


@pytest.mark.parametrize("previous", [1000, 999])
def test_legacy_near_equal_counts_are_recomputed_in_metadata_only(monkeypatch, tmp_path, previous):
    folder = staging(tmp_path, total=1000, active=previous)
    m = read_json(folder / "metadata.json")
    write_json(folder / "metadata.json", m | {"parameter_count_method": "non_lookup_parameters_v1"})
    original = (folder / "test.json.xz").read_bytes()
    resolve = Mock(return_value={"total_params": 1000, "active_params": 800,
                                "parameter_count_method": hp.METHOD,
                                "repo_id": "owner/model", "revision": "a" * 40})
    monkeypatch.setattr(hp, "resolve_model_counts", resolve)
    report = hp.enrich_metadata(tmp_path)
    assert report[0]["previous_active_params"] == previous
    assert read_json(folder / "metadata.json")["active_params"] == 800
    assert (folder / "test.json.xz").read_bytes() == original


def test_large_legacy_embedding_gap_is_preserved(monkeypatch, tmp_path):
    folder = staging(tmp_path, total=4628961024, active=1877497600)
    m = read_json(folder / "metadata.json")
    write_json(folder / "metadata.json", m | {"parameter_count_method": "non_lookup_parameters_v1"})
    original = (folder / "metadata.json").read_bytes()
    resolve = Mock()
    monkeypatch.setattr(hp, "resolve_model_counts", resolve)
    assert hp.enrich_metadata(tmp_path) == []
    resolve.assert_not_called()
    assert (folder / "metadata.json").read_bytes() == original


def test_alias_is_display_only_and_exact_mixed_revisions_are_not_guessed(monkeypatch, tmp_path):
    folder = staging(tmp_path, revision="unee", model_id="unee")
    resolve = resolver(monkeypatch)
    report = hp.enrich_metadata(tmp_path)
    assert report[0]["source"] == "linked-checkpoint"
    resolve.assert_called_once_with("owner/model", "main", api=None)
    # Restore missing counts and provide two explicitly evaluated revisions.
    m = read_json(folder / "metadata.json")
    write_json(folder / "metadata.json", m | {"total_params": None, "active_params": None,
                                              "parameter_count_method": None})
    for name, revision in [("test", "a" * 40), ("other", "b" * 40)]:
        write_json(folder / f"{name}.json.xz", {"model": {
            "id": "owner/model", "adapter": "test", "revision": revision}})
    resolve.reset_mock()
    assert hp.enrich_metadata(tmp_path)[0]["status"] == "unresolved"
    resolve.assert_not_called()
