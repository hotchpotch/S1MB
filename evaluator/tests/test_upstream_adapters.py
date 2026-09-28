"""Verify upstream boundaries without loading weights or running model inference."""

import json
from types import ModuleType, SimpleNamespace

import pytest

from s1mb.adapters.base import questions_for_api
from s1mb.adapters.decider import DeciderAdapter
from s1mb.adapters.jevforge import JevForgeAdapter
from s1mb.adapters.jevk5 import JevK5Adapter
from s1mb.adapters.minojev import MinojevAdapter
from s1mb.adapters.upstream import UpstreamAdapter, candidate_batches
from s1mb.data import Case


def cases():
    return Case.model_validate(
        {
            "case_id": "boundary",
            "group_id": "g",
            "language": "en",
            "state": {"evidence": "text"},
            "questions": [
                {
                    "id": "choose",
                    "task": "choice",
                    "system_prompt": "System.",
                    "instructions": "Choose.",
                    "options": [
                        {"id": "x", "description": "Same"},
                        {"id": "y", "description": "Same"},
                    ],
                },
                {
                    "id": "judge",
                    "task": "noul",
                    "instructions": "Judge.",
                    "options": [
                        {"id": "true", "description": "Yes"},
                        {"id": "false", "description": "No"},
                    ],
                },
                {
                    "id": "rate",
                    "task": "score",
                    "instructions": "Rate.",
                    "options": [
                        {"id": "low", "description": "Low", "value": 10},
                        {"id": "high", "description": "High", "value": 30},
                    ],
                },
            ],
            "targets": {
                "choose": {"kind": "hard_label", "probabilities": {"x": 1, "y": 0}},
                "judge": {"kind": "hard_label", "probabilities": {"true": 0, "false": 1}},
                "rate": {"kind": "hard_label", "probabilities": {"low": 1, "high": 0}},
            },
            "provenance": {"secret": "gold-must-not-be-sent"},
        }
    ).inference()


def answer(definition):
    if definition["type"] == "choice":
        return {"type": "choice", "probabilities": {"x": 0.2, "y": 0.8}}
    if definition["type"] == "noul":
        return {"type": "noul", "noul": 0.7}
    return {"type": "score", "probabilities": {"0": 0.3, "1": 0.7}}


@pytest.mark.parametrize("cls", [JevForgeAdapter, DeciderAdapter, JevK5Adapter])
def test_native_api_preserves_instructions_ids_and_score_scale(cls):
    case = cases()
    seen = []

    def decide(state, questions, **kwargs):
        assert state == (
            json.dumps(case.state, ensure_ascii=False) if cls is JevForgeAdapter else case.state
        )
        assert "gold-must-not-be-sent" not in str(questions)
        seen.append(questions)
        if cls is JevK5Adapter:
            return answer(questions)
        answers = {key: answer(value) for key, value in questions.items()}
        return {"answers": answers} if cls is DeciderAdapter else answers

    adapter = object.__new__(cls)
    adapter.engine = SimpleNamespace(decide=decide, system_one=decide)
    predictions = adapter.predict(case)
    assert len(seen) == 3
    sent = seen[0] if cls is JevK5Adapter else seen[0]["choose"]
    assert sent == questions_for_api(case.questions)["choose"]
    assert sent["instructions"] == "System.\n\nChoose."
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities["true"] == pytest.approx(0.7)
    assert predictions[2].probabilities == {"low": 0.3, "high": 0.7}


def test_no_automatic_cpu_fallback():
    adapter = UpstreamAdapter()
    with pytest.raises(ValueError, match="no CPU fallback"):
        adapter.setup("test", "unreachable/model", "main", "missing-source", "cpu")


def test_candidate_memory_budget_keeps_every_path_in_order():
    lengths = [200, 200, 3000, 5000, 100, 100]
    batches = list(candidate_batches(list(range(6)), lengths))
    assert [item for batch in batches for item in batch] == list(range(6))
    for batch in batches:
        assert len(batch) == 1 or max(lengths[i] for i in batch) * len(batch) <= 4096
    assert [3] in batches  # An oversized path is not truncated or dropped.


def test_minojev_accepts_native_fp32_error_but_rejects_invalid_distribution(monkeypatch):
    case = cases()
    case = case.model_copy(update={"questions": case.questions[:1]})
    adapter = object.__new__(MinojevAdapter)
    adapter.types = ModuleType("mock_minojev_types")
    monkeypatch.setattr(adapter.types, "request_from_object", lambda value: value, raising=False)
    adapter.options = None
    row = {"candidate_ids": ["x", "y"], "probabilities": [0.99809164, 0.00190824]}
    adapter.engine = SimpleNamespace(score=lambda *_: [row])
    assert sum(adapter.predict(case)[0].probabilities.values()) == pytest.approx(1)
    row["probabilities"] = [0.9, 0.2]
    with pytest.raises(ValueError, match="rounding tolerance"):
        adapter.predict(case)


def test_attention_change_must_be_effective():
    adapter = UpstreamAdapter()
    adapter.name = "luce"
    adapter.settings = {}
    adapter.attention_model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="eager"),
        set_attn_implementation=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="did not activate"):
        adapter.set_attention("sdpa")
    assert "attention_implementation" not in adapter.settings


def test_von_rejects_backend_that_cannot_represent_candidate_isolation():
    adapter = UpstreamAdapter()
    adapter.name = "von"
    with pytest.raises(ValueError, match="4D masks"):
        adapter.set_attention("flash_attention_2")


def test_von_literal_markers_are_not_candidate_delimiters():
    import re

    from s1mb.adapters.von import candidate_marker_positions

    def pack(state, question, options):
        prefix = f"{question} {state}".strip()
        return prefix + " [SEP] " + " ".join("[MASK] " + o.strip() for o in options)

    def tokenize(text, **_):
        # ModernBERT's AddedToken includes preceding whitespace in its offset.
        matches = list(re.finditer(r"\s?\[MASK\]", text))
        return {"input_ids": [99] * len(matches), "offset_mapping": [m.span() for m in matches]}

    network = SimpleNamespace(pack_sequence=pack, tokenizer=tokenize, mask_token_id=99)
    ids, positions = candidate_marker_positions(
        network, "Paper discusses [MASK].", "Explain [MASK].", ["Literal [MASK]", "Other"]
    )
    assert len(ids) == 5
    assert positions == [2, 4]
