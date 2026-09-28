"""Check scoring semantics and reject results that would corrupt a leaderboard."""

import copy

import pytest

from s1mb.adapters.base import decode_answers, questions_for_api
from s1mb.adapters.dummy import DummyAdapter
from s1mb.data import (
    DATA_DIR,
    Case,
    Prediction,
    Result,
    load_benchmark,
    load_cases,
    load_category,
    read_json,
    validate_result,
)
from s1mb.metrics import calculate
from s1mb.runner import evaluate


def example(task="choice"):
    ids = ["false", "true"] if task == "noul" else ["a", "b"]
    return Case.model_validate(
        {
            "case_id": "case",
            "group_id": "group",
            "language": "en",
            "state": {"text": "Input"},
            "questions": [
                {
                    "id": "q",
                    "task": task,
                    "instructions": "Judge.",
                    "options": [
                        {"id": ids[0], "description": "Low", "value": 10},
                        {"id": ids[1], "description": "High", "value": 30},
                    ],
                }
            ],
            "targets": {"q": {"kind": "soft", "probabilities": {ids[0]: 0.25, ids[1]: 0.75}}},
            "provenance": {"secret_label": "must not reach inference"},
        }
    )


@pytest.mark.parametrize(
    ("task", "expected"), [("choice", 0.75), ("noul", 0.0225), ("score", 0.15)]
)
def test_soft_target_metrics(task, expected):
    c = example(task)
    ids = [o.id for o in c.questions[0].options]
    p = Prediction(
        case_id="case", question_id="q", probabilities=dict(zip(ids, [0.4, 0.6], strict=True))
    )
    assert next(iter(calculate([c], [p], task).values())) == pytest.approx(expected)


def test_adapter_never_receives_targets_or_source_metadata():
    value = example().inference().model_dump()
    assert set(value) == {"case_id", "state", "questions"}
    assert "must not reach inference" not in str(value)


def test_choice_tie_uses_candidate_order():
    c = example()
    pred = Prediction(case_id="case", question_id="q", probabilities={"b": 0.5, "a": 0.5})
    assert calculate([c], [pred], "choice")["target_mass_at_prediction"] == 0.25


def test_score_api_indices_map_back_to_source_ids():
    c = example("score").inference()
    result = decode_answers(c, {"q": {"type": "score", "probabilities": {"0": 0.2, "1": 0.8}}})
    assert result[0].probabilities == {"a": 0.2, "b": 0.8}
    assert questions_for_api(c.questions)["q"]["criteria"] == ["Low", "High"]


def test_laya_rounding_is_bounded_not_arbitrary_normalization():
    c = example().inference()
    decoded = decode_answers(
        c, {"q": {"type": "choice", "probabilities": {"a": 0.5001, "b": 0.5}}}, rounded=True
    )
    assert decoded[0].probabilities is not None
    assert sum(decoded[0].probabilities.values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        decode_answers(
            c, {"q": {"type": "choice", "probabilities": {"a": 0.1, "b": 0.2}}}, rounded=True
        )


@pytest.mark.dataset
def test_fixed_dataset_inventory():
    category = load_category(DATA_DIR, "english-v1")
    counts = {"choice": 0, "noul": 0, "score": 0}
    for name in category.benchmarks:
        b = load_benchmark(DATA_DIR, name)
        counts[b.task] += b.decision_count
        assert len(load_cases(DATA_DIR, b)) == b.case_count
    assert len(category.benchmarks) == 137
    assert counts == {"choice": 8764, "noul": 12357, "score": 5148}


@pytest.mark.dataset
def test_result_validation_rejects_tampering_and_false_completion(tmp_path):
    b = load_benchmark(DATA_DIR, "arc-choice-test-v1")
    result = evaluate(DummyAdapter(), DATA_DIR, b, tmp_path, "test", limit=2)
    validate_result(DATA_DIR, result)
    assert result.status == "partial"
    source = read_json(DATA_DIR / "datasets/hub-source.json")
    assert result.environment["dataset_source"] == {
        "repo_id": source["repo_id"],
        "revision": source["revision"],
    }
    assert len(source["revision"]) == 40
    stale = copy.deepcopy(result)
    key = next(iter(stale.environment["input_hashes"]))
    stale.environment["input_hashes"][key] = "different-input-same-case-id"
    with pytest.raises(ValueError, match="input hashes"):
        validate_result(DATA_DIR, stale)
    del stale.environment["input_hashes"]
    with pytest.raises(ValueError, match="input hashes"):
        validate_result(DATA_DIR, stale)
    bad = copy.deepcopy(result)
    bad.status = "complete"
    with pytest.raises(ValueError, match="completion"):
        validate_result(DATA_DIR, bad)
    bad = copy.deepcopy(result)
    bad.metrics[b.primary_metric] = 0.12345
    with pytest.raises(ValueError, match="recomputed"):
        validate_result(DATA_DIR, bad)
    bad = copy.deepcopy(result)
    bad.predictions.append(bad.predictions[0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_result(DATA_DIR, bad)
    with pytest.raises(ValueError, match="already exists"):
        evaluate(DummyAdapter(), DATA_DIR, b, tmp_path, "test", limit=2)


@pytest.mark.dataset
def test_runner_records_failures_without_partial_adapter_output(tmp_path):
    class Broken(DummyAdapter):
        def predict(self, case):
            return []

    b = load_benchmark(DATA_DIR, "arc-choice-test-v1")
    result = evaluate(Broken(), DATA_DIR, b, tmp_path, "broken", limit=2)
    assert result.counts.failed == 2
    assert result.metrics[b.primary_metric] is None
    assert result.status == "partial"


@pytest.mark.dataset
def test_real_result_fixture():
    for path in (DATA_DIR / "results").rglob("*.json"):
        validate_result(DATA_DIR, Result.model_validate_json(path.read_text()))


def test_batch_runner_strips_targets_and_preserves_case_order():
    from s1mb.runner import predict_cases

    class Batched(DummyAdapter):
        case_batch_size = 2

        def __init__(self):
            self.sizes = []

        def predict_batch(self, cases):
            self.sizes.append(len(cases))
            for case in cases:
                assert not hasattr(case, "targets")
                assert not hasattr(case, "provenance")
            return [self.predict(case) for case in cases]

    cases = [example().model_copy(update={"case_id": str(i)}) for i in range(5)]
    adapter = Batched()
    outputs = list(predict_cases(adapter, cases))
    assert adapter.sizes == [2, 2, 1]
    assert [result[0].case_id for _, result in outputs] == [str(i) for i in range(5)]


def test_failed_batch_retries_cases_without_losing_successful_outputs():
    from s1mb.runner import predict_cases

    class Batched(DummyAdapter):
        case_batch_size = 3

        def predict_batch(self, cases):
            raise RuntimeError("batch failure")

        def predict(self, case):
            if case.case_id == "1":
                raise ValueError("case failure")
            return super().predict(case)

    cases = [example().model_copy(update={"case_id": str(i)}) for i in range(3)]
    outputs = list(predict_cases(Batched(), cases))
    assert outputs[0][1][0].case_id == "0"
    assert isinstance(outputs[1][1], ValueError)
    assert outputs[2][1][0].case_id == "2"


def test_optimized_ichi_rejects_cpu_before_loading_checkpoint():
    from s1mb.adapters.system_ichi import SystemIchiAdapter

    with pytest.raises(ValueError, match="requires CUDA"):
        SystemIchiAdapter("unused", "unused", "cpu")


@pytest.mark.parametrize("limit", [None, 1, 2])
def test_ichi_question_groups_pack_across_cases_without_splitting_decisions(limit):
    from types import ModuleType

    from s1mb.adapters.system_ichi import SystemIchiAdapter

    class Packer:
        @staticmethod
        def pack_decisions(items, budget, padded_documents):
            assert budget == 32_768 and padded_documents
            return [list(reversed(items))]

    adapter = object.__new__(SystemIchiAdapter)
    adapter.questions_per_call = limit
    adapter.batching = ModuleType("test_packer")
    adapter.batching.__dict__["pack_decisions"] = Packer.pack_decisions
    # These questions can come from different cases, including repeated IDs.
    prepared = [object() for _ in range(5)]
    groups = list(adapter._question_groups(prepared))
    assert {id(item) for group in groups for item in group} == {id(p) for p in prepared}
    assert sum(map(len, groups)) == 5
    assert all(len(group) <= (limit or 5) for group in groups)
    assert len(groups) == (1 if limit is None else (5 + limit - 1) // limit)


def test_bekko_native_input_preserves_structure_and_excludes_supervision():
    import json

    from s1mb.adapters.bekko import native_input

    case = example("noul")
    case.questions[0].instructions_json = '{"judge":["defect","denial"]}'
    case.questions[0].options[1].description_json = '["A defect", "An explicit denial"]'
    source = native_input(case.inference())
    assert set(source) == {"state_json", "decisions"}
    decision = source["decisions"][0]
    assert json.loads(decision["instructions_json"]) == {"judge": ["defect", "denial"]}
    assert json.loads(decision["criteria"][1]["description_json"]) == [
        "A defect",
        "An explicit denial",
    ]
    assert [c["id"] for c in decision["criteria"]] == ["false", "true"]
    assert "secret_label" not in json.dumps(source)
    assert "probabilities" not in json.dumps(source)
