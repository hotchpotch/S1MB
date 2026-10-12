"""Native block-attention selection and cache release for the 2B runtime."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.jev_style_2b import JevStyle2BAdapter
from s1mb.adapters.upstream import UpstreamAdapter


def test_native_block_backend_and_constructor_arguments(monkeypatch):
    calls = []

    def setup(self, *args):
        self.path = "checkpoint"
        self.settings = {}

    def engine(path, **kw):
        calls.append((path, kw))
        return SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="native-block")),
            default_temperature=0.828,
        )

    native = SimpleNamespace(
        JevStyleDecision=engine, CONTEXT_LIMIT=25600, BLOCK=2048, ATTN_NAME="native-block",
    )
    monkeypatch.setattr(UpstreamAdapter, "setup", setup)
    monkeypatch.setattr("s1mb.adapters.jev_style_2b.importlib.import_module", lambda name: native)
    adapter = JevStyle2BAdapter("unused", "main", "unused", "cuda:0")
    assert calls == [("checkpoint", {
        "device": "cuda:0", "dtype": "float32", "max_len": 25600,
        "verify": True, "keep_state": True, "cuda_graphs": False,
    })]
    assert adapter.settings["attention_implementation"] == "native-block"
    assert adapter.settings["temperature"] == 0.828


def test_cache_released_before_generic_cleanup(monkeypatch):
    adapter = JevStyle2BAdapter.__new__(JevStyle2BAdapter)
    events = []
    adapter.engine = SimpleNamespace(close=lambda: events.append("native-cache"))
    monkeypatch.setattr(UpstreamAdapter, "close", lambda self: events.append("generic"))
    adapter.close()
    assert events == ["native-cache", "generic"]


def test_invalid_budget_before_load():
    with pytest.raises(ValueError, match="positive"):
        JevStyle2BAdapter("unused", "main", "unused", "cuda:0", context_limit=0)
