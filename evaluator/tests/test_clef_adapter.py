"""Clef boundary checks without checkpoint downloads or GPU inference."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.clef import ClefAdapter, clef_record


def test_rendering_preserves_semantics_and_anonymizes_ids():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    case.questions[2].options.reverse()
    record, mappings = clef_record(case)
    assert record["state"] == case.state
    choice, noul, score = record["questions"].values()
    assert choice["instructions"] == {"system": "System.", "instruction": "Choose."}
    assert list(choice["criteria"].values()) == [{"nested": True}, "Same"]
    assert noul["criteria"] == {"true": "Yes", "false": "No"}
    assert score["criteria"] == [
        {"value": 30.0, "description": "High"},
        {"value": 10.0, "description": "Low"},
    ]
    assert mappings["field_000002"] == {"0": "high", "1": "low"}
    assert "gold-must-not-be-sent" not in str(record)
    assert "boundary" not in str(record)


class Values:
    def __init__(self, values):
        self.values = values

    def float(self):
        return self

    def softmax(self, dim):
        assert dim == -1
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def adapter(length=10, values=(0.3, 0.7)):
    model = ClefAdapter.__new__(ClefAdapter)
    encoded = SimpleNamespace(
        input_ids=tuple(range(length)),
        questions=[
            SimpleNamespace(
                question_id="field_000000", option_ids=("option_000000", "option_000001")
            ),
            SimpleNamespace(question_id="field_000001", option_ids=("true", "false")),
            SimpleNamespace(question_id="field_000002", option_ids=("0", "1")),
        ],
    )

    def encode(tokenizer, record, max_length):
        assert max_length > length
        return encoded

    model.native = SimpleNamespace(encode_record=encode, collate_records=lambda *args: None)
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    model.torch = cast(Any, SimpleNamespace(inference_mode=nullcontext, device=lambda x: x))
    model.device = "cuda"
    model.context_limit = 10
    model.engine = lambda batch: [[Values(list(values)) for _ in range(3)]]
    return model


def test_native_probability_alignment():
    predictions = adapter().predict(cases())
    assert predictions[0].probabilities == {"x": 0.3, "y": 0.7}
    assert predictions[1].probabilities == {"true": 0.3, "false": 0.7}
    assert predictions[2].probabilities == {"low": 0.3, "high": 0.7}


def test_overflow_precedes_inference():
    model = adapter(length=11)
    model.engine = lambda batch: pytest.fail("Inference must not run")
    with pytest.raises(ValueError, match="refusing truncation"):
        model.predict(cases())


@pytest.mark.parametrize("values", [(float("nan"), 0.7), (-0.1, 1.1), (0.2, 0.3), (1.0,)])
def test_invalid_probabilities_fail(values):
    with pytest.raises(ValueError):
        adapter(values=values).predict(cases())


def test_configuration_rejects_cpu_and_invalid_limit_before_download():
    with pytest.raises(ValueError, match="explicit CUDA"):
        ClefAdapter("Cloudflare/clef", "main", "cpu")
    with pytest.raises(ValueError, match="positive"):
        ClefAdapter("Cloudflare/clef", "main", "cuda", context_limit=0)


def test_metadata_and_cleanup(monkeypatch):
    model = adapter()
    model.name = "clef"
    model.model_id = "Cloudflare/clef"
    model.revision = "a" * 40
    model.source_digest = "b" * 64
    model.settings = {"input_length_policy": "reject-overflow", "source_revision": model.revision}
    monkeypatch.setattr("s1mb.adapters.upstream.parameter_metadata", lambda engine: {})
    monkeypatch.setattr("s1mb.adapters.upstream.importlib.metadata.version", lambda name: "test")
    info = model.metadata()
    assert info.revision == "a" * 40
    assert info.settings["source_revision"] == info.revision
    assert info.settings["input_length_policy"] == "reject-overflow"
    cleared = []
    model.torch.cuda = SimpleNamespace(empty_cache=lambda: cleared.append(True))
    model.close()
    assert cleared == [True]
    assert not hasattr(model, "engine")


def test_cuda_without_index_resolves_to_first_visible_device(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "s1mb.adapters.clef.checkpoint_path", lambda model, revision: (tmp_path, "a" * 40)
    )
    devices = []

    def setup(self, name, model, revision, source, device):
        devices.append(device)
        raise RuntimeError("stop before runtime import")

    monkeypatch.setattr(ClefAdapter, "setup", setup)
    with pytest.raises(RuntimeError, match="stop before runtime import"):
        ClefAdapter("Cloudflare/clef", "main", "cuda")
    assert devices == ["cuda:0"]


def test_initialization_uses_native_language_model_and_pinned_code(monkeypatch, tmp_path):
    revision = "a" * 40
    monkeypatch.setattr(
        "s1mb.adapters.clef.checkpoint_path", lambda model, rev: (tmp_path, revision)
    )
    backbone = object()
    engine = SimpleNamespace(language_model=backbone)
    processor = object()
    loads = []

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return engine, processor

    native = SimpleNamespace(load_release_model=load)
    spec = SimpleNamespace(
        name="fake_clef_runtime", loader=SimpleNamespace(exec_module=lambda m: None)
    )
    monkeypatch.setattr(
        "s1mb.adapters.clef.importlib.util.spec_from_file_location", lambda *a: spec
    )
    monkeypatch.setattr("s1mb.adapters.clef.importlib.util.module_from_spec", lambda s: native)
    monkeypatch.setitem(__import__("sys").modules, spec.name, native)

    def setup(self, name, model, rev, source, device):
        assert rev == revision
        assert source == str(tmp_path)
        assert device == "cuda:0"
        self.torch = cast(Any, SimpleNamespace(bfloat16="bf16"))

    monkeypatch.setattr(ClefAdapter, "setup", setup)
    accelerated = []
    monkeypatch.setattr(ClefAdapter, "enable_kernels", lambda self: accelerated.append(True))
    model = ClefAdapter("Cloudflare/clef", "main", "cuda", context_limit=32768)
    assert model.attention_model is backbone
    assert model.engine is engine
    assert model.processor is processor
    assert model.settings["source_revision"] == revision
    assert model.settings["max_input_tokens"] == 32768
    assert loads == [
        (tmp_path, {"device": "cuda:0", "dtype": "bf16", "attn_implementation": "sdpa"})
    ]
    assert accelerated == [True]
