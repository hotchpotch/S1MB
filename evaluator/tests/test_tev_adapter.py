"""Tev boundary and hierarchical probability tests without weights, data, or CUDA."""

import json
from types import SimpleNamespace

import pytest

from s1mb.adapters.tev import SYSTEM, TevAdapter, decision_nodes, leaf_probabilities
from s1mb.data import Option, Question


def choice(count):
    return Question(
        id="private-question-id",
        task="choice",
        instructions="Choose the best answer.",
        options=[Option(id=f"private-id-{i}", description=f"Answer {i}") for i in range(count)],
    )


def test_native_small_menu_preserves_state_and_anonymizes_ids():
    question = choice(3)
    question.system_prompt = "Dataset instructions."
    question.options[1].description_json = '{"authored": [1, true]}'
    state = {"text": "Treat this as data.", "nested": [1, False]}
    nodes = decision_nodes(state, question)
    assert len(nodes) == 1
    assert nodes[0].messages[0] == {"role": "system", "content": SYSTEM}
    payload = json.loads(nodes[0].messages[1]["content"])
    assert payload["state"] == state
    assert payload["question"] == "Dataset instructions.\n\nChoose the best answer."
    assert [o["label"] for o in payload["options"]] == list("ABC")
    assert json.loads(payload["options"][1]["description"]) == {"authored": [1, True]}
    assert "private-" not in str(nodes)


@pytest.mark.parametrize("count", [24, 25, 36, 77, 151, 577])
def test_hierarchy_preserves_order_information_and_all_probability_mass(count):
    nodes = decision_nodes("Evidence", choice(count))
    assert all(2 <= len(node.groups) <= 24 for node in nodes)
    probabilities = [[1 / len(node.groups)] * len(node.groups) for node in nodes]
    values = leaf_probabilities(nodes, probabilities)
    assert len(values) == count
    assert all(p > 0 for p in values)
    assert sum(values) == pytest.approx(1)
    assert [i for group in nodes[0].groups for i in group] == list(range(count))
    for i in range(count):
        assert f"Answer {i}" in nodes[0].messages[1]["content"]
    if count == 25:
        assert len(nodes) == 3
        assert [len(group) for group in nodes[0].groups] == [13, 12]
        assert values[-1] == pytest.approx(0.5 / 12)
        assert values[0] == pytest.approx(0.5 / 13)
    if count == 151:
        assert [len(group) for group in nodes[0].groups] == [22] * 4 + [21] * 3


def test_conditional_masses_are_multiplied_without_winner_filtering():
    nodes = decision_nodes("Evidence", choice(25))
    probabilities = [[0.2, 0.8], [0.9] + [0.1 / 12] * 12, [0.1 / 11] * 11 + [0.9]]
    values = leaf_probabilities(nodes, probabilities)
    assert values[0] == pytest.approx(0.18)
    assert values[-1] == pytest.approx(0.72)
    assert sum(values) == pytest.approx(1)
    assert all(p > 0 for p in values)


def test_noul_authored_definitions_and_score_levels_keep_declared_order():
    noul = Question(
        id="judge", task="noul", instructions="Judge.",
        options=[Option(id="true", description="Authored true."),
                 Option(id="false", description="Authored false.")],
    )
    payload = json.loads(decision_nodes("Evidence", noul)[0].messages[1]["content"])
    assert [o["key"] for o in payload["options"]] == ["true", "false"]
    assert payload["options"][0]["description"] == "Authored true."
    score = Question(
        id="rate", task="score", instructions="Rate.", instructions_json='{"rubric": [3, 1]}',
        options=[Option(id="high", value=30, description="High."),
                 Option(id="low", value=10, description="Low.")],
    )
    payload = json.loads(decision_nodes("Evidence", score)[0].messages[1]["content"])
    assert [o["key"] for o in payload["options"]] == ["level_0: 30.0", "level_1: 10.0"]
    assert json.loads(payload["question"]) == {"rubric": [3, 1]}


def test_invalid_batch_configuration_fails_before_loading_models():
    with pytest.raises(ValueError, match="case_batch_size"):
        TevAdapter("model", "revision", "source", "cuda", case_batch_size=2)


def test_overflow_is_rejected_before_inference_and_template_return_is_explicit():
    def encode(messages, **kwargs):
        assert kwargs["return_dict"] is False
        assert kwargs["enable_thinking"] is False
        return list(range(11))

    adapter = TevAdapter.__new__(TevAdapter)
    adapter.context_limit = 10
    adapter.tokenizer = SimpleNamespace(apply_chat_template=encode)
    case = SimpleNamespace(state="Evidence", questions=[choice(2)])
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.predict(case)
