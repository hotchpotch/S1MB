"""RSI rejects a truncation report rather than saving altered-input results."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.rsi import RSIAdapter
from s1mb.data import InferenceCase


def test_unexpected_native_truncation_report_is_rejected(monkeypatch):
    case = InferenceCase.model_validate({
        "case_id": "sample", "state": "state",
        "questions": [{"id": "pick", "task": "choice", "instructions": "Choose",
                       "options": [{"id": "a", "description": "First"},
                                   {"id": "b", "description": "Second"}]}],
    })
    adapter = RSIAdapter.__new__(RSIAdapter)
    monkeypatch.setattr(adapter, "wire",
                        SimpleNamespace(state_to_text=lambda state: state, RequestError=ValueError),
                        raising=False)
    adapter.engine = SimpleNamespace(request=lambda *a: {"truncated": {"tokens": 1}})
    with pytest.raises(ValueError, match="truncation"):
        adapter.predict(case)


def test_constructor_uses_native_server_default_before_applying_budget(monkeypatch):
    import os
    from dataclasses import dataclass

    @dataclass
    class Encoder:
        max_length: int = 32768
        truncate: str = "none"

    def native_decider(*args, **kwargs):
        assert not os.environ.get("RSIJEV_MAX_LENGTH")
        assert not os.environ.get("RSIJEV_TRUNCATE")
        return SimpleNamespace(
            served=SimpleNamespace(enc=Encoder(), meta={"spec": {}}), calibration="native",
        )

    def setup(self, *args):
        self.path = "checkpoint"
        self.settings = {}

    monkeypatch.setenv("RSIJEV_MAX_LENGTH", "2048")
    monkeypatch.setenv("RSIJEV_TRUNCATE", "left")
    monkeypatch.setattr(RSIAdapter, "setup", setup)
    monkeypatch.setattr("s1mb.adapters.rsi.importlib.import_module",
                        lambda name: SimpleNamespace(Decider=native_decider))
    adapter = RSIAdapter("model", "revision", "source", "cuda:0", context_limit=8192)
    assert adapter.engine.served.enc.max_length == 8192
    assert adapter.engine.served.enc.truncate == "none"
    assert os.environ["RSIJEV_MAX_LENGTH"] == "2048"
    assert os.environ["RSIJEV_TRUNCATE"] == "left"
