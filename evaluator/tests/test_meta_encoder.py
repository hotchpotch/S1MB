"""Verify Meta Encoder's released text-choice input boundary."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_hf_data import native_row

from s1mb import cli
from s1mb.adapters.meta_encoder import (
    MetaEncoderAdapter,
    chat_messages,
    render_candidate,
    render_query,
)
from s1mb.hf_data import case_from_row


def test_renderer_preserves_raw_json_declared_order_and_option_id_candidates():
    row = native_row()
    case = case_from_row(row).inference()
    choice, noul, score = case.questions

    expected = (
        "Select the correct option. {\"messages\":[{\"role\":\"user\","
        "\"content\":\"A sample\"}],\"query\":\"Question\"}\n\n"
        "Task: {'task': 'Choose a label', 'constraints': ['Use context']}\n"
        "Criteria:\n- b: {'label': 'B'}\n- a: A\nOptions: b, a"
    )
    assert render_query(case, choice) == expected
    assert [render_candidate(option) for option in choice.options] == ["b", "a"]
    assert [render_candidate(option) for option in noul.options] == ["false", "true"]
    assert [render_candidate(option) for option in score.options] == ["high-id", "low-id"]
    assert "teacher-only" not in render_query(case, score)


def test_renderer_falls_back_to_compact_state_and_sanitizes_chat_text():
    row = native_row()
    case = case_from_row(row).inference().model_copy(
        update={"state": {"video": "<|video|>", "patch": "<|patch|>"}, "state_json": None}
    )
    query = render_query(case, case.questions[1])
    assert query.startswith('Select the correct option. {"video":"<|video|>","patch":"<|patch|>"}')
    messages = chat_messages(query)
    assert messages[0]["content"][0]["text"] == "Represent the user's input."
    text = messages[1]["content"][0]["text"]
    assert "<|video|>" not in text and "<|patch|>" not in text
    assert "\n" not in text and "  " not in text


@pytest.mark.parametrize(
    "device,temperature,context_limit,attention,message",
    [
        ("cpu", 0.03, None, "sdpa", "explicit CUDA"),
        ("cuda:0", 0.0, None, "sdpa", "finite and positive"),
        ("cuda:0", float("nan"), None, "sdpa", "finite and positive"),
        ("cuda:0", 0.03, 0, "sdpa", "context limit"),
        ("cuda:0", 0.03, None, "eager", "attention"),
    ],
)
def test_constructor_rejects_invalid_runtime_before_loading(
    device, temperature, context_limit, attention, message
):
    with pytest.raises(ValueError, match=message):
        MetaEncoderAdapter(
            "facebook/meta-encoder",
            "revision",
            device,
            temperature=temperature,
            context_limit=context_limit,
            attention=attention,
        )


def test_encode_rejects_complete_overflow_before_model_forward():
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "chat"

        def __call__(self, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": SimpleNamespace(shape=(1, 3))}

    adapter = MetaEncoderAdapter.__new__(MetaEncoderAdapter)
    adapter.processor = cast(Any, Processor())
    adapter.context_limit = 2
    adapter.engine = cast(Any, SimpleNamespace())
    with pytest.raises(ValueError, match="refusing truncation"):
        adapter._encode(["text"], 1)


def test_fixture_state_json_remains_valid_and_target_free():
    case = case_from_row(native_row()).inference()
    assert case.state_json is not None
    assert json.loads(case.state_json) == case.state
    assert set(case.model_dump()) == {"case_id", "state", "state_json", "questions"}


@pytest.mark.parametrize(
    "adapter,extra,message",
    [
        ("meta-encoder", [], "temperature is required"),
        ("meta-encoder", ["--temperature", "nan"], "finite and positive"),
        ("dummy", ["--temperature", "0.03"], "applies only to Meta Encoder"),
    ],
)
def test_cli_requires_scoped_explicit_temperature(
    monkeypatch, capsys, tmp_path, adapter, extra, message
):
    data = Path(__file__).resolve().parents[1] / "data"
    args = [
        "s1mb",
        "--data-dir",
        str(data),
        "run",
        "--adapter",
        adapter,
        "--offline-dataset",
        "--category",
        "smoke-v1",
        "--output",
        str(tmp_path),
    ]
    if adapter != "dummy":
        args.extend(["--model", "facebook/meta-encoder", "--device", "cuda:0"])
    args.extend(extra)
    monkeypatch.setattr("sys.argv", args)

    def unexpected_dataset_session(*args, **kwargs):
        pytest.fail("Invalid CLI options must be rejected before acquiring the dataset lock")

    monkeypatch.setattr("s1mb.dataset_source.dataset_session", unexpected_dataset_session)
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert message in capsys.readouterr().err
