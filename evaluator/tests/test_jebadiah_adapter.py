"""Checkpoint loading must not apply LoRA twice to merged Jebadiah releases."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from s1mb.adapters.jebadiah import JebadiahAdapter, jebadiah_questions


@pytest.mark.parametrize("merged", [False, True])
def test_jebadiah_loads_checkpoint_tokenizer_and_saved_temperatures(tmp_path, monkeypatch, merged):
    base = "Qwen/Qwen3.5-9B"
    revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    if merged:
        (tmp_path / "jebadiah.json").write_text(json.dumps({"base": f"{base}@{revision}"}))
    else:
        (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
        revision = "68c46c4b3498877f3ef123c856ecfde50c39f404"
    temperatures = {"choice": 1.1863, "noul": 1.0903, "score": 0.8329}
    native = SimpleNamespace(
        FP32_CANDIDATE_LOGITS=True,
        load_tokenizer=Mock(return_value=object()),
        load_base=Mock(return_value=object()),
        load_adapter=Mock(return_value=object()),
        read_temperatures=Mock(return_value=temperatures),
        Scorer=Mock(return_value=SimpleNamespace(temperatures=temperatures)),
    )

    def setup(self, *args):
        self.path = tmp_path
        self.torch = SimpleNamespace(bfloat16="bfloat16")

    monkeypatch.setattr(JebadiahAdapter, "setup", setup)
    monkeypatch.setattr(JebadiahAdapter, "enable_kernels", lambda self: None)
    monkeypatch.setitem(sys.modules, "jebadiah_model", native)
    adapter = JebadiahAdapter("model", "pinned", "source", "cuda:0")
    load_path = str(tmp_path) if merged else base
    load_revision = None if merged else revision
    native.load_tokenizer.assert_called_once_with(load_path, load_revision)
    native.load_base.assert_called_once_with(
        load_path, load_revision, attn_implementation="sdpa", dtype="bfloat16", device="cuda:0"
    )
    if merged:
        native.load_adapter.assert_not_called()
        model = native.load_base.return_value
    else:
        native.load_adapter.assert_called_once_with(native.load_base.return_value, str(tmp_path))
        model = native.load_adapter.return_value
    native.read_temperatures.assert_called_once_with(str(tmp_path))
    native.Scorer.assert_called_once_with(
        model,
        native.load_tokenizer.return_value,
        max_tokens=32768,
        temperatures=temperatures,
        device="cuda:0",
    )
    assert adapter.settings["base_revision"] == revision
    assert adapter.settings["checkpoint_format"] == (
        "merged-bfloat16" if merged else "lora-adapter"
    )
    assert adapter.settings["temperatures"] == temperatures


def test_jebadiah_retains_structured_criteria_numeric_values_and_authored_order():
    from test_upstream_adapters import cases

    case = cases()
    case.questions[0].options[0].description_json = '{"nested":true}'
    case.questions[1].options.reverse()
    case.questions[2].options.reverse()
    rendered = jebadiah_questions(case)
    assert json.loads(rendered["choose"]["criteria"]["option_0"]) == {"nested": True}
    assert list(rendered["judge"]["criteria"]) == ["false", "true"]
    assert [json.loads(value)["value"] for value in rendered["rate"]["criteria"].values()] == [30, 10]
    assert "gold-must-not-be-sent" not in str(rendered)
