"""Native typed-contract preservation, answer alignment and overflow errors."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.decision2 import Decision2Adapter, Decision2InputTooLong, decision2_questions


def test_structured_anonymous_choice_authored_noul_numeric_score():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    case.questions[2].options.reverse()
    questions = decision2_questions(case)
    assert questions["choose"]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["criteria"] == [
        {"value": 30.0, "description": "High"},
        {"value": 10.0, "description": "Low"},
    ]
    assert "gold-must-not-be-sent" not in str(questions)


def test_single_question_calls_use_anonymous_transport_id_and_native_probabilities():
    adapter = Decision2Adapter.__new__(Decision2Adapter)

    def system_one(*, state, questions):
        assert state == cases().state
        assert set(questions) == {"decision"}
        question = questions["decision"]
        task = question["type"]
        if task == "noul":
            answer = {"type": task, "noul": 0.7}
        else:
            keys = list(question["criteria"]) if task == "choice" else ["0", "1"]
            answer = {"type": task, "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}
        return {"answers": {"decision": answer}}

    adapter.engine = SimpleNamespace(system_one=system_one)
    results = adapter.predict(cases())
    assert results[0].probabilities == {"x": 0.2, "y": 0.8}
    assert results[1].probabilities["true"] == 0.7
    assert results[2].probabilities == {"low": 0.2, "high": 0.8}


@pytest.mark.parametrize(
    "answer",
    [None, {"other": {}}, {"decision": {"type": "choice", "error": "invalid_model_output"}}],
)
def test_missing_malformed_and_native_error_answers_rejected(answer):
    adapter = Decision2Adapter.__new__(Decision2Adapter)
    adapter.engine = SimpleNamespace(system_one=lambda **kwargs: {"answers": answer})
    with pytest.raises(ValueError):
        adapter.predict(cases())


def test_native_overflow_has_distinct_saved_error_type():
    adapter = Decision2Adapter.__new__(Decision2Adapter)
    adapter.engine = SimpleNamespace(
        system_one=lambda **kwargs: {
            "answers": {"decision": {"type": "choice", "error": "max_length_exceeded"}},
        }
    )
    with pytest.raises(Decision2InputTooLong):
        adapter.predict(cases())


def test_non_mapping_native_answer_rejected():
    adapter = Decision2Adapter.__new__(Decision2Adapter)
    adapter.engine = SimpleNamespace(
        system_one=lambda **kwargs: {"answers": {"decision": []}}
    )
    with pytest.raises(TypeError, match="malformed answer"):
        adapter.predict(cases())


def test_invalid_budget_before_checkpoint_load():
    with pytest.raises(ValueError, match="positive"):
        Decision2Adapter("unused", "main", "unused", "cuda:0", context_limit=0)


def test_hub_links_are_independent_files_before_native_verification(tmp_path, monkeypatch):
    from s1mb.adapters import decision2

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"pinned-weights")
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "config.json").write_text("{}")
    copied = []

    def setup(self, *args):
        self.path = snapshot
        self.settings = {}
        self.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))

    def load(path, **kwargs):
        root = Path(path)
        assert not any(p.is_symlink() for p in root.rglob("*"))
        assert (root / "model.safetensors").read_bytes() == b"pinned-weights"
        (root / "model.safetensors").write_bytes(b"independent-copy")
        copied.append(root)
        return SimpleNamespace(
            max_input_tokens=8192,
            backend=SimpleNamespace(
                model=SimpleNamespace(backbone=SimpleNamespace(
                    config=SimpleNamespace(_attn_implementation="sdpa")
                )),
                temperatures={"choice": 1}, score_bias=None,
            ),
            manifest={"profile": "qwen-full", "identity": {}},
        )

    original_import = decision2.importlib.import_module
    monkeypatch.setattr(Decision2Adapter, "setup", setup)
    monkeypatch.setattr(
        decision2.importlib, "import_module",
        lambda name: SimpleNamespace(Decision2=SimpleNamespace(from_pretrained=load))
        if name == "decision2" else original_import(name),
    )
    adapter = Decision2Adapter("unused", "pinned", "unused", "cuda:0")
    assert blob.read_bytes() == b"pinned-weights"
    adapter.close()
    assert not copied[0].exists()
