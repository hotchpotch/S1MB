"""Xor metadata preserves parameter provenance within the result schema."""

from s1mb.adapters.xor import XorAdapter


def test_metadata_excludes_hub_lookup_fields():
    adapter = object.__new__(XorAdapter)
    adapter.model_id = "example/xor"
    adapter.name = "xor"
    adapter.device = "cuda:0"
    adapter.source_digest = "source-sha"
    adapter.settings = {}
    adapter.counts = {
        "repo_id": "example/xor", "subfolder": None, "revision": "pinned-sha",
        "total_params": 100, "active_params": 80,
        "parameter_count_method": "embedding_excluded_parameters_v1",
    }
    metadata = adapter.metadata()
    assert metadata.revision == "pinned-sha"
    assert metadata.total_params == 100
    assert metadata.active_params == 80
    assert metadata.parameter_count_method == "embedding_excluded_parameters_v1"
