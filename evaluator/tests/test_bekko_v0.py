"""Portable adapter contract tests without model packages, GPU, or private data."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from s1mb.adapters.bekko_v0 import BekkoV0Adapter
from s1mb.data import InferenceCase


def case(name="case", task="choice"):
    ids = ["false", "true"] if task == "noul" else ["b", "a"]
    return InferenceCase.model_validate(
        {
            "case_id": name,
            "state": {"text": name},
            "questions": [
                {
                    "id": "decision",
                    "task": task,
                    "instructions": "Judge.",
                    "options": [
                        {"id": key, "description": "one two", "value": i * 4}
                        for i, key in enumerate(ids)
                    ],
                }
            ],
        }
    )


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    calls = []

    class Runtime(SimpleNamespace):
        pass

    runtime = Runtime(
        tasks=["choice", "noul", "score"],
        attn_implementation="sdpa",
        query_length=32,
        document_length=8,
        context_length=128,
        encoder=SimpleNamespace(
            backbone=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128))
        ),
        prefix_layout="instruction_state",
        query_truncation="right",
        task_token_ids={},
        budget_policy="adaptive-v1",
        settings={"query_length": 32},
    )

    class Model:
        def eval(self):
            return self

        def __getitem__(self, index):
            return runtime

        def predict(self, requests, **kwargs):
            if not requests:
                return []
            calls.append((requests, kwargs))
            return [
                {
                    d["id"]: {
                        "probabilities": {
                            c["id"]: p for c, p in zip(d["criteria"], [0.25, 0.75], strict=True)
                        }
                    }
                    for d in request["decisions"]
                }
                for request in requests
            ]

    def load_model(*args, **kwargs):
        assert kwargs["attn_implementation"] == "auto"
        return Model()

    original_import = importlib.import_module
    monkeypatch.setattr(
        "s1mb.adapters.bekko_v0.importlib.import_module",
        lambda name: (
            SimpleNamespace(get_num_threads=lambda: 4)
            if name == "torch"
            else SimpleNamespace(
                get_class_from_dynamic_module=lambda *a, **k: load_model
            )
            if name == "transformers.dynamic_module_utils"
            else original_import(name)
        ),
    )
    monkeypatch.setattr("s1mb.adapters.bekko_v0.parameter_metadata", lambda model: {})
    monkeypatch.setattr(
        "s1mb.adapters.bekko_v0.resolve_checkpoint", lambda *a: (tmp_path, "owner/model", "a" * 40)
    )
    value = BekkoV0Adapter("owner/model", "cpu", cpu_smoke=True, case_batch_size=2)
    return value, calls


def test_portable_batch_preserves_case_task_candidate_order(adapter):
    value, calls = adapter
    cases = [case("first"), case("second", "noul"), case("third", "score")]
    outputs = value.predict_batch(cases)
    assert [[p.case_id for p in row] for row in outputs] == [["first"], ["second"], ["third"]]
    assert list(outputs[0][0].probabilities) == ["b", "a"]
    assert outputs[1][0].probabilities == {"false": 0.25, "true": 0.75}
    assert outputs[2][0].probabilities == {"b": 0.25, "a": 0.75}
    assert [len(groups) for groups, _ in calls] == [2, 1]
    assert all(
        options["token_budget"] == 64000 and not options["show_progress_bar"]
        for _, options in calls
    )
    assert value.predict_batch([]) == []
    value.close()  # CPU cleanup must never access CUDA.


def test_long_input_delegates_budgeting_to_native_runtime(adapter):
    value, calls = adapter
    value.query_length = 8
    value.document_length = 2
    value.predict(case("long input " * 100))
    assert len(calls) == 1
    assert calls[0][1]["query_length"] == 8
    assert calls[0][1]["document_length"] == 2
    assert calls[0][1]["context_length"] == 128


def test_cpu_requires_explicit_smoke():
    with pytest.raises(ValueError, match="smoke"):
        BekkoV0Adapter("unused", "cpu")


def test_cli_rejects_cpu_full_evaluation(monkeypatch, capsys):
    from s1mb.cli import main

    monkeypatch.setattr("sys.argv", ["s1mb", "run", "--adapter", "bekko-v0", "--device", "cpu"])
    monkeypatch.setattr("s1mb.dataset_source.dataset_session", lambda *a, **k: nullcontext())
    monkeypatch.setattr("s1mb.cli.load_category", lambda *a: SimpleNamespace(benchmarks=[]))
    with pytest.raises(SystemExit):
        main()
    assert "CPU requires" in capsys.readouterr().err


def test_hub_resolution_pins_snapshot_to_resolved_sha(tmp_path, monkeypatch):
    from s1mb.adapters.bekko_v0 import resolve_checkpoint

    calls = []
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        lambda: SimpleNamespace(model_info=lambda repo, revision: SimpleNamespace(sha="a" * 40)),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, revision: calls.append((repo, revision)) or str(tmp_path),
    )
    path, model_id, sha = resolve_checkpoint("owner/model", "main")
    assert (path, model_id, sha) == (tmp_path, "owner/model", "a" * 40)
    assert calls == [("owner/model", "a" * 40)]
    with pytest.raises(ValueError, match="Hugging Face model ID"):
        resolve_checkpoint(str(tmp_path), "main")


def test_metadata_records_selected_attention_backend(adapter, monkeypatch):
    value, _ = adapter
    monkeypatch.setattr("s1mb.adapters.bekko_v0.importlib.metadata.version", lambda name: "test")
    value.runtime.attn_implementation = "flash_attention_2"
    assert value.metadata().settings["attention"] == "flash_attention_2"
