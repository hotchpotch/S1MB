"""Portable adapter contract tests without model packages, GPU, or private data."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from s1mb.adapters.bekko_v0 import BekkoV0Adapter, validate_groups
from s1mb.data import InferenceCase


class Tokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs.get("truncation") is False
        return {"input_ids": [text.split() for text in texts]}


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
        query_length=32,
        document_length=8,
        prefix_layout="instruction_state",
        query_truncation="right",
        task_token_ids={},
        tokenizer=Tokenizer(),
        settings={"query_length": 32},
    )

    class Model:
        def eval(self):
            return self

        def __getitem__(self, index):
            return runtime

        def predict_groups(self, groups, **kwargs):
            if not groups:
                return []
            calls.append((groups, kwargs))
            return [SimpleNamespace(tolist=lambda: [0.25, 0.75]) for _ in groups]

    def render(value, **kwargs):
        assert set(value) == {"state_json", "decisions"}
        return [
            SimpleNamespace(
                query=value["state_json"],
                candidates=["one two" for _ in d["criteria"]],
                task=d["type"],
                metadata=SimpleNamespace(candidate_ids=tuple(c["id"] for c in d["criteria"])),
            )
            for d in value["decisions"]
        ]

    monkeypatch.setattr(importlib.import_module(__name__), "input_groups", render, raising=False)
    monkeypatch.setattr(
        "s1mb.adapters.bekko_v0.load_runtime",
        lambda p: SimpleNamespace(
            BekkoSentenceTransformer=lambda *a, **k: Model(), input_groups=render
        ),
    )
    original_import = importlib.import_module
    monkeypatch.setattr(
        "s1mb.adapters.bekko_v0.importlib.import_module",
        lambda name: (
            SimpleNamespace(get_num_threads=lambda: 4) if name == "torch" else original_import(name)
        ),
    )
    monkeypatch.setattr("s1mb.adapters.bekko_v0.parameter_metadata", lambda model: {})
    value = BekkoV0Adapter(str(tmp_path), "cpu", cpu_smoke=True, case_batch_size=2)
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


def test_portable_overflow_rejected_before_prediction(adapter):
    value, calls = adapter
    value.document_length = 2
    with pytest.raises(ValueError, match="refusing truncation"):
        value.predict(case())
    assert calls == []


def test_task_marker_reserves_an_extra_token():
    runtime = SimpleNamespace(
        query_truncation="right", task_token_ids={"choice": 10}, tokenizer=Tokenizer()
    )
    group = SimpleNamespace(query="short", candidates=["one two three"], task="choice")
    with pytest.raises(ValueError, match="refusing truncation"):
        validate_groups(runtime, [group], 8, 4)
    group.task = "score"
    validate_groups(runtime, [group], 8, 4)


def test_balanced_preflight_includes_field_boundaries():
    runtime = SimpleNamespace(query_truncation="balanced", task_token_ids={}, tokenizer=Tokenizer())
    group = SimpleNamespace(
        task="choice",
        candidates=["short"],
        query_parts=SimpleNamespace(system="", instruction="one two", context="three four"),
    )
    with pytest.raises(ValueError, match="Balanced query"):
        validate_groups(runtime, [group], 7, 8)
    validate_groups(runtime, [group], 8, 8)


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
