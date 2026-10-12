"""Verify target-free reasoning, fixed verification and bounded artifacts."""

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_hf_data import native_row

from s1mb import cli
from s1mb.adapters.meta_encoder import MetaEncoderAdapter, _parsed_text, render_query
from s1mb.adapters.meta_encoder_think import (
    REASONER_ID,
    REASONER_REVISION,
    MetaEncoderThinkAdapter,
    composite_parameter_metadata,
    contextual_query,
    load_reasoning_contexts,
    parse_reasoning_generation,
    reasoner_prompt,
    reasoning_context,
    validate_reasoning_payload,
    verification_probabilities,
    verification_queries,
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


def test_verification_preserves_option_order_and_shared_reasoning():
    case = case_from_row(native_row()).inference()
    question = case.questions[0]
    record = {"status": "usable", "truncated_reasoning": "Shared evidence.", "raw_generation": "raw"}
    requests = verification_queries(case, question, record)
    assert len(requests) == len(question.options)
    assert all(text.startswith("Reasoning context:\nShared evidence.\n\n") for text in requests)
    assert len({text.rsplit("\nProposed answer:", 1)[0] for text in requests}) == 1
    for text, option in zip(requests, question.options, strict=True):
        description = _parsed_text(option.description_json, option.description)
        assert text.endswith(f"Proposed answer: {option.id}: {description}")
        assert "teacher-only" not in text


def test_verification_is_fixed_one_hot_over_correct_minus_incorrect_gaps():
    assert verification_probabilities([[0.4, 0.1], [0.5, 0.4]]) == [1.0, 0.0]
    assert verification_probabilities([[0.1, 0.4], [0.4, 0.1]], 0) == [0.0, 1.0]


def test_verification_ties_select_first_declared_option():
    assert verification_probabilities([[0.5, 0.2], [0.5, 0.2]]) == [1.0, 0.0]


@pytest.mark.parametrize("temperature", [0.03, -1, float("nan"), float("inf")])
def test_verification_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        verification_probabilities([[0.5, 0.2]], temperature)


@pytest.mark.parametrize("pairs", [[], [[0.5]], [[float("nan"), 0.2]], [[float("inf"), 0.2]]])
def test_verification_rejects_malformed_or_nonfinite_gaps(pairs):
    with pytest.raises(ValueError):
        verification_probabilities(pairs)


def valid_payload():
    return {
        "format_version": 1,
        "reasoner": {"id": REASONER_ID, "revision": REASONER_REVISION},
        "records": [{
            "case_id": "case", "question_id": "question", "query_sha256": "a" * 64,
            "status": "usable", "truncated_reasoning": "Evidence.",
            "raw_generation": "Evidence.\nFINAL_ANSWER: b",
        }],
    }


def test_context_digest_identifies_exact_consumed_bytes(tmp_path):
    path = tmp_path / "contexts.json"
    payload = valid_payload()
    first = json.dumps(payload).encode()
    path.write_bytes(first)
    loaded, digest = load_reasoning_contexts(path)
    assert loaded == payload
    assert digest == hashlib.sha256(first).hexdigest()
    payload["records"][0]["truncated_reasoning"] = "Different evidence."
    path.write_text(json.dumps(payload))
    assert load_reasoning_contexts(path)[1] != digest


def test_context_size_is_checked_before_json_decode(tmp_path, monkeypatch):
    path = tmp_path / "contexts.json"
    path.write_bytes(b"not JSON but oversized")
    monkeypatch.setattr("s1mb.adapters.meta_encoder_think.MAX_CONTEXT_FILE_BYTES", 5)
    with pytest.raises(ValueError, match="size limit"):
        load_reasoning_contexts(path)


@pytest.mark.parametrize("field,value", [
    ("case_id", 1), ("question_id", None), ("query_sha256", "not-a-hash"),
    ("status", []), ("truncated_reasoning", 123), ("raw_generation", None),
    ("truncated_reasoning", "x" * 131073), ("truncated_reasoning", ""),
])
def test_context_record_fields_are_validated(field, value):
    payload = valid_payload()
    payload["records"][0][field] = value
    with pytest.raises(ValueError, match="record"):
        validate_reasoning_payload(payload)


@pytest.mark.parametrize("payload", [[], {}, {"format_version": True}, {
    "format_version": 1, "reasoner": [],
}])
def test_context_schema_is_validated(payload):
    with pytest.raises((TypeError, ValueError)):
        validate_reasoning_payload(payload)


def test_context_record_count_is_bounded(monkeypatch):
    payload = valid_payload()
    payload["records"] *= 2
    monkeypatch.setattr("s1mb.adapters.meta_encoder_think.MAX_CONTEXT_RECORDS", 1)
    with pytest.raises(ValueError, match="records"):
        validate_reasoning_payload(payload)


@pytest.mark.parametrize("version", [1, 2])
def test_supported_context_versions_preserve_exact_artifact_identity(tmp_path, version):
    payload = valid_payload()
    payload["format_version"] = version
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps(payload))
    loaded, digest = load_reasoning_contexts(path)
    assert loaded["format_version"] == version
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_invalid_context_is_rejected_before_model_loading(tmp_path, monkeypatch):
    path = tmp_path / "contexts.json"
    path.write_text("not JSON")

    def unexpected_loading(*args, **kwargs):
        pytest.fail("model loading must follow context validation")

    monkeypatch.setattr(MetaEncoderAdapter, "__init__", unexpected_loading)
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        MetaEncoderThinkAdapter(reasoning_contexts=path, scoring="candidate_verify")


def test_verification_constructor_defaults_to_zero_and_records_artifact(tmp_path, monkeypatch):
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps(valid_payload()))

    def fake_init(self, *args, temperature, **kwargs):
        assert temperature == 0
        self.temperature = temperature

    monkeypatch.setattr(MetaEncoderAdapter, "__init__", fake_init)
    monkeypatch.setattr(MetaEncoderAdapter, "metadata", lambda self: ModelInfo(
        id="encoder", adapter="meta-encoder", revision="revision", settings={},
    ))
    adapter = MetaEncoderThinkAdapter(reasoning_contexts=path, scoring="candidate_verify")
    settings = adapter.metadata().settings
    assert settings["reasoning_context_artifact"] == {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "format_version": 1,
    }
    assert settings["verification_tie_break"] == "first-declared-option"
    assert settings["probability_rule"] == "one-hot-verification-argmax"
    assert str(path) not in json.dumps(settings)


@pytest.fixture
def context_generator():
    path = Path(__file__).resolve().parents[1] / "scripts/generate_meta_encoder_think_contexts.py"
    spec = importlib.util.spec_from_file_location("think_context_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atomic_publication_is_complete_and_does_not_overwrite(tmp_path, context_generator):
    path = tmp_path / "contexts.json"
    context_generator.publish_contexts(path, valid_payload())
    assert load_reasoning_contexts(path)[0] == valid_payload()
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        context_generator.publish_contexts(path, valid_payload())
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_failed_publication_leaves_no_partial_artifact(tmp_path, context_generator, monkeypatch):
    path = tmp_path / "contexts.json"

    def fail(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(context_generator.os, "link", fail)
    with pytest.raises(OSError, match="publication failure"):
        context_generator.publish_contexts(path, valid_payload())
    assert list(tmp_path.iterdir()) == []


def test_failed_serialization_leaves_no_partial_artifact(tmp_path, context_generator, monkeypatch):
    path = tmp_path / "contexts.json"

    def fail(payload, stream, **kwargs):
        stream.write("{partial")
        raise ValueError("simulated serialization failure")

    monkeypatch.setattr(context_generator.json, "dump", fail)
    with pytest.raises(ValueError, match="serialization failure"):
        context_generator.publish_contexts(path, valid_payload())
    assert list(tmp_path.iterdir()) == []


def test_verification_predictions_preserve_native_option_ids_for_all_tasks():
    case = case_from_row(native_row()).inference()
    winning_ids = {question.options[-1].id for question in case.questions}

    class CosinePairs:
        def __init__(self, pairs):
            self.pairs = pairs

        def __matmul__(self, other):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.pairs

    adapter = cast(Any, MetaEncoderThinkAdapter.__new__(MetaEncoderThinkAdapter))
    adapter.scoring, adapter.temperature, adapter.context_limit = "candidate_verify", 0, 32768
    adapter.torch = SimpleNamespace(stack=CosinePairs)
    adapter._contexts = {
        (case.case_id, question.id, hashlib.sha256(render_query(case, question).encode()).hexdigest()): {
            "status": "usable", "truncated_reasoning": "Evidence.", "raw_generation": "Evidence.",
        } for question in case.questions
    }
    adapter.processor = lambda **kwargs: {"input_ids": [[1]]}
    adapter._chat = lambda text: text
    adapter._candidate_embeddings = lambda texts: SimpleNamespace(T=None)

    def encode(requests, batch_size):
        return [
            [0.4, 0.1] if text.rsplit("\nProposed answer: ", 1)[1].split(":", 1)[0] in winning_ids
            else [0.1, 0.4] for text in requests
        ]

    adapter._encode = encode
    predictions = adapter.predict(case)
    assert len(predictions) == 3
    for question, prediction in zip(case.questions, predictions, strict=True):
        ids = [option.id for option in question.options]
        assert prediction.probabilities is not None
        assert list(prediction.probabilities) == ids
        assert list(prediction.probabilities.values()) == [0.0, 1.0]


@pytest.mark.parametrize("temperature", [None, 0])
def test_cli_verification_accepts_zero_or_no_temperature(temperature, monkeypatch):
    command = ["run", "--adapter", "meta-encoder-think", "--model", "facebook/meta-encoder",
               "--device", "cuda:0", "--think-scoring", "candidate_verify",
               "--reasoning-contexts", "contexts.json"]
    if temperature is not None:
        command += ["--temperature", str(temperature)]
    captured = []
    monkeypatch.setattr("sys.argv", ["s1mb", *command])
    monkeypatch.setattr(cli, "execute", lambda args, parser: captured.append(args))
    monkeypatch.setattr("s1mb.dataset_source.dataset_session", lambda *args, **kwargs: nullcontext())
    cli.main()
    assert captured[0].temperature == 0


def test_cli_verification_rejects_nonzero_temperature_before_execution(monkeypatch, capsys):
    command = ["s1mb", "run", "--adapter", "meta-encoder-think", "--model", "facebook/meta-encoder",
               "--device", "cuda:0", "--think-scoring", "candidate_verify",
               "--reasoning-contexts", "contexts.json", "--temperature", "0.03"]
    monkeypatch.setattr("sys.argv", command)
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert "fixed at zero" in capsys.readouterr().err


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
