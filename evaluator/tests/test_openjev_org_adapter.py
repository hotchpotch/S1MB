"""OpenJev organization model boundary tests without model weights or CUDA."""

import math
from types import SimpleNamespace

import pytest

from s1mb.adapters.openjev_org import (
    NOUL_TEMPERATURE,
    OpenJevOrgAdapter,
    calibrated_probabilities,
    decision_options,
    render_prompt,
)
from s1mb.data import InferenceCase, Option, Question


def question(task="choice"):
    return Question(
        id="private-question",
        task=task,
        instructions="Decide.",
        options=[
            Option(id="private-a", description="First"),
            Option(id="private-b", description="Second"),
        ],
    )


def test_structured_native_prompt_excludes_identifiers():
    q = question()
    q.instructions_json = '{"rules": [true, 1]}'
    q.system_prompt = "Authored system"
    q.options[0].description_json = '{"nested": [false, 2]}'
    instruction, options, ids = decision_options(q)
    prompt = render_prompt({"document": [1, True]}, instruction, options)
    assert ids == ["private-a", "private-b"]
    assert "private" not in prompt
    assert prompt.startswith('State:\n{"document": [1, true]}\n\nQuestion: ')
    assert "{'system': 'Authored system', 'instruction': {'rules': [True, 1]}}" in prompt
    assert '[A] option_0: {"nested": [false, 2]}' in prompt
    assert "[B] option_1: Second" in prompt


def test_authored_noul_definitions_and_calibration():
    q = Question(
        id="judge",
        task="noul",
        instructions="Judge.",
        options=[
            Option(id="false", description="Authored negative."),
            Option(id="true", description="Authored positive."),
        ],
    )
    _, options, ids = decision_options(q)
    assert ids == ["true", "false"]
    assert options == [("yes", "Authored positive."), ("no", "Authored negative.")]
    values = calibrated_probabilities([0.8, 0.2], "noul")
    assert values[0] == pytest.approx(1 / (1 + math.exp(-math.log(4) / NOUL_TEMPERATURE)))
    assert sum(values) == pytest.approx(1)
    q.options[0].description = ""
    assert decision_options(q)[1][1] == ("no", "The statement is false.")
    assert 0 < calibrated_probabilities([0, 1], "noul")[0] < 0.5


def test_numeric_score_order_and_prediction_inverse_mapping(monkeypatch):
    q = Question(
        id="rate",
        task="score",
        instructions="Rate.",
        options=[
            Option(id="high", value=30, description="High"),
            Option(id="low", value=-5, description="Low"),
            Option(id="middle", value=2, description="Middle"),
        ],
    )
    instruction, options, ids = decision_options(q)
    assert instruction.endswith("Rate along the ordered levels below (lowest first).")
    assert ids == ["low", "middle", "high"]
    assert options == [("0", "Low"), ("1", "Middle"), ("2", "High")]
    adapter = OpenJevOrgAdapter.__new__(OpenJevOrgAdapter)
    monkeypatch.setattr(adapter, "distribution", lambda *args: [0.2, 0.3, 0.5])
    result = adapter.predict(InferenceCase(case_id="private-case", state="Text", questions=[q]))
    assert result[0].probabilities == {"low": 0.2, "middle": 0.3, "high": 0.5}
    assert [o.value for o in q.options] == [30, -5, 2]


@pytest.mark.parametrize("count", [53, 77, 151, 577])
def test_chunk_winner_anchor_recovers_global_distribution(count, monkeypatch):
    adapter = OpenJevOrgAdapter.__new__(OpenJevOrgAdapter)
    calls = []

    def readout(state, instruction, options):
        assert 1 <= len(options) <= 52
        calls.append(options)
        weights = [float(description) for _, description in options]
        return [value / sum(weights) for value in weights]

    monkeypatch.setattr(adapter, "readout", readout)
    options = [(str(i), str(i + 1)) for i in range(count)]
    values = adapter.distribution("state", "instruction", options)
    assert values == pytest.approx([(i + 1) / sum(range(1, count + 1)) for i in range(count)])
    assert [option for call in calls[:-1] for option in call] == options
    assert len(calls[-1]) == math.ceil(count / 52)


def test_overflow_fails_before_inference_with_thinking_disabled():
    def encode(messages, **kwargs):
        assert kwargs["return_dict"] is False
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return list(range(10))

    adapter = OpenJevOrgAdapter.__new__(OpenJevOrgAdapter)
    adapter.context_limit = 10
    adapter.tokenizer = SimpleNamespace(apply_chat_template=encode)
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter.readout("text", "question", [("a", "A"), ("b", "B")])


@pytest.mark.parametrize("values", [[float("nan"), 0], [1.1, -0.1], [0.1, 0.1]])
def test_malformed_probabilities_are_rejected(values):
    with pytest.raises(ValueError):
        calibrated_probabilities(values, "choice")


def test_model_errors_are_not_replaced_with_uniform_probabilities(monkeypatch):
    adapter = OpenJevOrgAdapter.__new__(OpenJevOrgAdapter)

    def fail(*args):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(adapter, "distribution", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        adapter.predict(InferenceCase(case_id="case", state="text", questions=[question()]))


def test_invalid_context_fails_before_model_loading():
    with pytest.raises(ValueError, match="context_limit"):
        OpenJevOrgAdapter("model", "sha", "source", "cuda", context_limit=20000)
