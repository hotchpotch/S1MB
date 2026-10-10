"""Verify the public MetaEncoder-think preprocessing and calibration contract."""

import hashlib
from pathlib import Path

import pytest
from test_hf_data import native_row

from s1mb import cli
from s1mb.adapters.meta_encoder import render_query
from s1mb.adapters.meta_encoder_think import (
    MetaEncoderThinkAdapter,
    composite_parameter_metadata,
    contextual_query,
    parse_reasoning_generation,
    reasoner_prompt,
    reasoning_context,
    recalibrate_probabilities,
)
from s1mb.data import ModelInfo
from s1mb.hf_data import case_from_row


def test_reasoner_prompt_wraps_the_unchanged_official_query_without_targets():
    case = case_from_row(native_row()).inference()
    query = render_query(case, case.questions[0])
    prompt = reasoner_prompt(query)
    assert prompt.endswith(query)
    assert "FINAL_ANSWER: <option ID>" in prompt
    assert "teacher-only" not in prompt


def test_boundary_policy_prefers_the_earliest_answer_like_line():
    raw = "First inspect the evidence.\nTherefore the answer is b.\nFINAL_ANSWER: b"
    parsed = parse_reasoning_generation(raw)
    assert parsed["status"] == "usable"
    assert parsed["cut_rule"] == "answer_like_conclusion"
    assert parsed["truncated_reasoning"] == "First inspect the evidence."
    assert parsed["answer_like_conclusion_before_marker"] is True


def test_missing_boundary_falls_back_to_the_entire_nonempty_generation():
    raw = "Inspect the evidence without a detected conclusion."
    parsed = parse_reasoning_generation(raw)
    record = {**parsed, "raw_generation": raw}
    assert parsed["status"] == "missing_boundary"
    assert reasoning_context(record) == raw
    assert contextual_query("original", record) == f"Reasoning context:\n{raw}\n\noriginal"


def test_empty_fallback_is_rejected():
    with pytest.raises(ValueError, match="No generated reasoning context"):
        reasoning_context({"status": "missing_boundary", "raw_generation": ""})


def test_probability_recalibration_preserves_order_sum_and_argmax():
    original = {"b": 0.7, "a": 0.2, "c": 0.1}
    calibrated = recalibrate_probabilities(original, 0.03, 0.018467)
    assert list(calibrated) == list(original)
    assert sum(calibrated.values()) == pytest.approx(1.0)
    assert max(calibrated, key=calibrated.__getitem__) == "b"
    expected_ratio = (original["b"] / original["a"]) ** (0.03 / 0.018467)
    assert calibrated["b"] / calibrated["a"] == pytest.approx(expected_ratio)


def test_composite_parameter_counts_account_for_reasoner_and_encoder():
    encoder = ModelInfo(
        id="encoder",
        adapter="meta-encoder",
        revision="revision",
        total_params=10,
        active_params=8,
        parameter_count_method="embedding_excluded_parameters_v1",
    )
    reasoner = {
        "parameters": {
            "total_params": 20,
            "active_params": 17,
            "parameter_count_method": "embedding_excluded_parameters_v1",
        }
    }
    assert composite_parameter_metadata(encoder, reasoner) == {
        "total_params": 30,
        "active_params": 25,
        "parameter_count_method": "embedding_excluded_parameters_v1",
    }


@pytest.mark.parametrize("probabilities", [{"a": 1.0, "b": 0.0}, {"a": float("nan")}])
def test_probability_recalibration_rejects_nonpositive_or_nonfinite_sources(probabilities):
    with pytest.raises(ValueError, match="strictly positive"):
        recalibrate_probabilities(probabilities, 0.03, 0.02)


def test_adapter_looks_up_context_by_query_identity():
    case = case_from_row(native_row()).inference()
    question = case.questions[0]
    query = render_query(case, question)
    record = {
        "status": "usable",
        "truncated_reasoning": "A target-free rationale.",
        "raw_generation": "A target-free rationale.\nFINAL_ANSWER: b",
    }
    adapter = MetaEncoderThinkAdapter.__new__(MetaEncoderThinkAdapter)
    adapter._contexts = {
        (case.case_id, question.id, hashlib.sha256(query.encode()).hexdigest()): record
    }
    assert adapter._render_query(case, question) == (
        f"Reasoning context:\nA target-free rationale.\n\n{query}"
    )


def test_cli_requires_reasoning_contexts_before_dataset_access(monkeypatch, capsys, tmp_path):
    data = Path(__file__).resolve().parents[1] / "data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "s1mb",
            "--data-dir",
            str(data),
            "run",
            "--adapter",
            "meta-encoder-think",
            "--model",
            "facebook/meta-encoder",
            "--device",
            "cuda:0",
            "--temperature",
            "0.03",
        ],
    )

    def unexpected_dataset_session(*args, **kwargs):
        pytest.fail("invalid CLI options must be rejected before dataset access")

    monkeypatch.setattr("s1mb.dataset_source.dataset_session", unexpected_dataset_session)
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert "--reasoning-contexts is required" in capsys.readouterr().err
