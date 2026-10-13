"""Exercise released AutoJev interfaces without loading model weights."""

import json
from types import SimpleNamespace

import pytest

from s1mb.adapters.autojev import AutoJevAdapter


@pytest.mark.parametrize("text_only", [False, True])
def test_native_attention_metadata(monkeypatch, tmp_path, text_only):
    config = {"base_model": "base", "revision": "a" * 40, "text_only": text_only}
    (tmp_path / "decision_config.json").write_text(json.dumps(config))
    engine = SimpleNamespace(
        backbone=SimpleNamespace(config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(max_position_embeddings=32768),
        )),
        codes=["A", "B"], temperature=1.3,
    )
    if not text_only:
        engine.attention_mode = "noncausal_full_attention"
        engine.pooling = "mean"
    engine.eval = lambda: engine
    native = SimpleNamespace(MAX_OPTIONS=255, DecisionModel=lambda **kwargs: engine)

    def setup(self, *args):
        self.path = tmp_path

    monkeypatch.setattr("s1mb.adapters.autojev.sys.version_info", (3, 12))
    monkeypatch.setattr("s1mb.adapters.autojev.importlib.import_module", lambda name: native)
    monkeypatch.setattr(AutoJevAdapter, "setup", setup)
    monkeypatch.setattr(AutoJevAdapter, "enable_kernels", lambda self: None)
    adapter = AutoJevAdapter("model", "a" * 40, str(tmp_path), "cuda:0")
    assert adapter.settings["attention_mode"] == (
        "causal" if text_only else "noncausal_full_attention"
    )
    assert adapter.settings["pooling"] == ("last" if text_only else "mean")
    assert adapter.settings["temperature"] == 1.3
