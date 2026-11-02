"""Exercise released AutoJev interfaces without loading model weights."""

import json
from types import SimpleNamespace

import pytest

from s1mb.adapters.autojev import AutoJevAdapter


def load_adapter(monkeypatch, tmp_path, config, **engine_fields):
    config = {"base_model": "base", "revision": "a" * 40, **config}
    (tmp_path / "decision_config.json").write_text(json.dumps(config))
    engine = SimpleNamespace(
        backbone=SimpleNamespace(config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(max_position_embeddings=32768),
        )),
        codes=["A", "B"], temperature=1.3, **engine_fields,
    )
    engine.eval = lambda: engine
    native = SimpleNamespace(MAX_OPTIONS=255, DecisionModel=lambda **kwargs: engine)

    def setup(self, *args):
        self.path = tmp_path

    monkeypatch.setattr("s1mb.adapters.autojev.sys.version_info", (3, 12))
    monkeypatch.setattr("s1mb.adapters.autojev.importlib.import_module", lambda name: native)
    monkeypatch.setattr(AutoJevAdapter, "setup", setup)
    monkeypatch.setattr(AutoJevAdapter, "enable_kernels", lambda self: None)
    return AutoJevAdapter("model", "a" * 40, str(tmp_path), "cuda:0")


@pytest.mark.parametrize("text_only", [False, True])
def test_native_attention_metadata(monkeypatch, tmp_path, text_only):
    fields = {} if text_only else {"attention_mode": "noncausal_full_attention", "pooling": "mean"}
    adapter = load_adapter(monkeypatch, tmp_path, {"text_only": text_only}, **fields)
    assert adapter.settings["attention_mode"] == (
        "causal" if text_only else "noncausal_full_attention"
    )
    assert adapter.settings["pooling"] == ("last" if text_only else "mean")
    assert adapter.settings["temperature"] == 1.3


@pytest.mark.parametrize(("config", "engine_fields", "expected"), [
    ({"prompt_attention": "bidirectional"}, {}, "bidirectional"),
    ({"prompt_attention": "causal"}, {}, "causal"),
    ({"prompt_attention": "causal"}, {"prompt_attention": "bidirectional"}, "bidirectional"),
])
def test_text_only_records_effective_prompt_attention(
    monkeypatch, tmp_path, config, engine_fields, expected,
):
    adapter = load_adapter(monkeypatch, tmp_path, {"text_only": True, **config}, **engine_fields)
    assert adapter.settings["attention_mode"] == expected
