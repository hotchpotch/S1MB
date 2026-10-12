"""Complete-prefix admission and native candidate telemetry alignment."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.lfm_pcd import LFMPCDAdapter, admit_complete_input


def fake_engine():
    def constrained(context, schema, mode):
        assert mode == "token"
        labels = schema["properties"]["decision"]["enum"]
        assert "gold-must-not-be-sent" not in context + str(schema)
        return {"mode": mode, "calibrated": False, "fields": {"decision": {"candidates": [
            {"value": value, "probability": probability}
            for value, probability in zip(labels, [0.2, 0.8], strict=True)
        ]}}}

    return SimpleNamespace(
        compile=lambda schema, mode: SimpleNamespace(fields=[SimpleNamespace(suffix=(1, 2, 3))]),
        tokenizer=None, limits=None, constrained=constrained,
    )


def test_complete_input_budget_includes_field_suffix():
    native = SimpleNamespace(prompt_tokens=lambda *args: [1, 2])
    engine = fake_engine()
    admit_complete_input(engine, native, "context", {}, 5)
    with pytest.raises(ValueError, match="refusing truncation"):
        admit_complete_input(engine, native, "context", {}, 4)


def test_native_telemetry_maps_all_task_options():
    adapter = LFMPCDAdapter.__new__(LFMPCDAdapter)
    adapter.context_limit = 5
    adapter.engine = fake_engine()
    adapter.native = SimpleNamespace(prompt_tokens=lambda *args: [1, 2])
    predictions = adapter.predict(cases())
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities == {"true": 0.2, "false": 0.8}
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}


def test_overflow_rejected_before_constrained_forward():
    adapter = LFMPCDAdapter.__new__(LFMPCDAdapter)
    adapter.context_limit = 4
    adapter.engine = fake_engine()
    adapter.engine.constrained = lambda *args, **kw: pytest.fail("forward must not run")
    adapter.native = SimpleNamespace(prompt_tokens=lambda *args: [1, 2])
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(cases())


def test_candidate_reordering_rejected():
    adapter = LFMPCDAdapter.__new__(LFMPCDAdapter)
    adapter.context_limit = 5
    adapter.engine = fake_engine()
    real = adapter.engine.constrained

    def reordered(*args, **kw):
        response = real(*args, **kw)
        response["fields"]["decision"]["candidates"].reverse()
        return response

    adapter.engine.constrained = reordered
    adapter.native = SimpleNamespace(prompt_tokens=lambda *args: [1, 2])
    with pytest.raises(ValueError, match="different candidate"):
        adapter.predict(cases())


def test_invalid_budget_before_loading():
    with pytest.raises(ValueError, match="positive"):
        LFMPCDAdapter("unused", "main", "unused", "cuda:0", context_limit=0)


@pytest.mark.parametrize("message", [
    "request exceeds conservative GPU memory budget; reduce prompt or branch batch size",
    "prefix and suffix exceed model context",
])
def test_memory_retry_preserves_native_request_and_other_errors(monkeypatch, message):
    from contextlib import nullcontext

    from s1mb.adapters import lfm_pcd

    calls = []
    released = []

    def constrained(context, schema, mode):
        calls.append((context, schema, mode))
        if len(calls) == 1:
            raise ValueError(message)
        return "native-result"

    engine = SimpleNamespace(constrained=constrained, device="cuda:0")
    cuda = SimpleNamespace(
        device=lambda device: nullcontext(), empty_cache=lambda: released.append(True),
    )
    monkeypatch.setattr(lfm_pcd.importlib, "import_module", lambda name: SimpleNamespace(cuda=cuda))
    if message.startswith("request exceeds"):
        assert lfm_pcd.constrained_with_memory_retry(engine, "state", {}) == "native-result"
        assert calls == [("state", {}, "token"), ("state", {}, "token")]
        assert released == [True]
    else:
        with pytest.raises(ValueError, match="model context"):
            lfm_pcd.constrained_with_memory_retry(engine, "state", {})
        assert len(calls) == 1
        assert released == []


def test_persistent_native_memory_rejection_is_not_hidden(monkeypatch):
    from contextlib import nullcontext

    from s1mb.adapters import lfm_pcd

    calls = []
    released = []
    message = "request exceeds conservative GPU memory budget; reduce prompt or branch batch size"

    def constrained(*args, **kwargs):
        calls.append(True)
        raise ValueError(message)

    cuda = SimpleNamespace(
        device=lambda device: nullcontext(), empty_cache=lambda: released.append(True),
    )
    monkeypatch.setattr(lfm_pcd.importlib, "import_module", lambda name: SimpleNamespace(cuda=cuda))
    engine = SimpleNamespace(constrained=constrained, device="cuda:0")
    with pytest.raises(ValueError, match="conservative GPU memory"):
        lfm_pcd.constrained_with_memory_retry(engine, "state", {})
    assert len(calls) == 2
    assert released == [True]
