"""Exercise structured HF judgments, rendering alignment, and target isolation."""

import copy
import json
from pathlib import Path
from typing import cast

import pytest
from datasets import load_from_disk

from s1mb.data import DATA_DIR, load_benchmark, load_cases
from s1mb.hf_data import case_from_row


def native_row():
    return json.loads((Path(__file__).parent / "fixtures/system_one.json").read_text())


def test_native_structured_values_and_target_ids():
    row = native_row()
    case = case_from_row(row)
    assert case.state["messages"][0]["content"] == "A sample"
    assert case.questions[0].instructions == (
        '{"constraints":["Use context"],"task":"Choose a label"}'
    )
    assert case.questions[0].options[0].description == '{"label":"B"}'
    assert case.questions[2].options[1].description == '["High","Detailed"]'
    assert case.targets["quality"].probabilities == {"low-id": 0.75, "high-id": 0.25}
    assert [(o.id, o.value) for o in case.questions[2].options] == [("high-id", 0), ("low-id", 4)]
    assert case.questions[0].system_prompt == "Classify."


def test_jev_preserves_structured_input_and_authored_criteria():
    from s1mb.adapters.base import questions_for_api

    row = native_row()
    case = case_from_row(row)
    request = questions_for_api(case.questions, sort_score=True, structured=True)
    for decision in row["input"]["decisions"]:
        question = request[decision["id"]]
        instruction = json.loads(decision["instructions_json"])
        if decision["system_prompt"]:
            instruction = {"system": decision["system_prompt"], "instruction": instruction}
        assert question["instructions"] == instruction
        criteria = decision["criteria"]
        if decision["type"] == "score":
            assert question["criteria"] == [
                json.loads(c["description_json"])
                for c in sorted(criteria, key=lambda c: c["value"])
            ]
        else:
            assert question["criteria"] == {
                c["id"]: json.loads(c["description_json"]) for c in criteria
            }


def test_labels_and_auxiliary_text_do_not_change_inference():
    row = native_row()
    baseline = case_from_row(row).inference()
    row["targets"][0]["probabilities"] = [0.1, 0.9]
    row["targets"][0]["metadata_json"] = '{"answer":"teacher-secret"}'
    row["private_note"] = "must never enter model input"
    assert case_from_row(row).inference() == baseline
    assert set(baseline.model_dump()) == {"case_id", "state", "questions"}


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_decision",
        "missing_target",
        "duplicate_target",
        "target_ids",
        "probability",
        "ranking",
        "target_kind",
    ],
)
def test_malformed_alignment_is_rejected(kind):
    row = native_row()
    if kind == "duplicate_decision":
        row["input"]["decisions"].append(copy.deepcopy(row["input"]["decisions"][0]))
    elif kind == "missing_target":
        row["targets"].pop()
    elif kind == "duplicate_target":
        row["targets"].append(copy.deepcopy(row["targets"][0]))
    elif kind == "target_ids":
        row["targets"][0]["ids"] = ["low-id", "low-id"]
    elif kind == "probability":
        row["targets"][0]["probabilities"] = [-1, 2]
    elif kind == "ranking":
        row["input"]["decisions"][0]["kind"] = "ranking"
    elif kind == "target_kind":
        row["targets"][0]["kind"] = "ranking_distribution"
    with pytest.raises(ValueError):
        case_from_row(row)


@pytest.mark.dataset
def test_release_effective_instructions_and_soft_labels():
    row = load_from_disk(str(DATA_DIR / "datasets/laya__typed_decisions"))["test"][0]
    case = case_from_row(row)
    for q, d in zip(case.questions, row["input"]["decisions"], strict=True):
        assert q.instructions == json.loads(d["instructions_json"])
        assert q.system_prompt == d["system_prompt"]
        assert {o.id: o.value for o in q.options} == {c["id"]: c["value"] for c in d["criteria"]}
    assert case.state == json.loads(row["input"]["state_json"])
    assert any(0 < p < 1 for t in case.targets.values() for p in t.probabilities.values())


@pytest.mark.dataset
def test_nano_pointwise_format_and_original_order():
    b = load_benchmark(DATA_DIR, "synthetic-relevance--nanobeir--nanomsmarco-score-test-v1")
    cases = load_cases(DATA_DIR, b)
    assert len(cases) == 400
    assert cases[0].questions[0].id == "relevance_score"
    assert {o.value for o in cases[0].questions[0].options} == {0, 1, 2, 3, 4}
    row = load_from_disk(str(DATA_DIR / b.dataset))["test"][0]
    decision = next(d for d in row["input"]["decisions"] if d["id"] == "relevance_score")
    order = [c["id"] for c in decision["criteria"]]
    assert [o.id for o in cases[0].questions[0].options] == order


@pytest.mark.dataset
def test_mixed_dataset_filters_questions_and_targets_by_task():
    for task in ("choice", "noul", "score"):
        b = load_benchmark(DATA_DIR, f"laya--typed-decisions-{task}-test-v1")
        cases = load_cases(DATA_DIR, b)
        assert len(cases) == 100
        assert sum(len(c.questions) for c in cases) == b.decision_count
        assert all(all(q.task == task for q in c.questions) for c in cases)
        assert all(set(c.targets) == {q.id for q in c.questions} for c in cases)


@pytest.mark.dataset
def test_retrieval_noul_utility_is_not_normalized_score():
    rows = load_from_disk(str(DATA_DIR / "datasets/synthetic_relevance__nanobeir__NanoMSMARCO"))
    differences = []
    for row in rows["test"]:
        case = case_from_row(row)
        questions = {q.id: q for q in case.questions}
        assert set(questions) == {"relevance_score", "relevance_noul"}
        score = questions["relevance_score"]
        assert [o.value for o in score.options] == [0, 1, 2, 3, 4]
        target = case.targets[score.id].probabilities
        expected = sum(cast(float, o.value) * target[o.id] for o in score.options) / 4
        differences.append(abs(case.targets["relevance_noul"].probabilities["true"] - expected))
    assert max(differences) > 0.1
