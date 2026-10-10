"""DM-JEPA lossless rendering and overflow boundaries without GPU inference."""

import json
from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.dm_jepa import DMJEPAAdapter, dm_jepa_inputs, encode_inputs


class Formatter:
    @staticmethod
    def format_state_prompt(state, instructions):
        return json.dumps([state, instructions])

    @staticmethod
    def format_option_prompt(label, criteria):
        return (label, criteria)


def test_structured_values_and_declared_criterion_order():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    state, options = dm_jepa_inputs(case, case.questions[0], Formatter)
    assert json.loads(options[0][1]) == {"nested": True}
    assert options[0][0] == "option_0"
    assert "gold-must-not-be-sent" not in state
    assert "boundary" not in state
    _, options = dm_jepa_inputs(case, case.questions[1], Formatter)
    assert options == [("true", "Yes"), ("false", "No")]
    case.questions[2].options.reverse()
    _, options = dm_jepa_inputs(case, case.questions[2], Formatter)
    assert options == [("30.0", "High"), ("10.0", "Low")]


@pytest.mark.parametrize("state_length,option_length", [(17, 5), (16, 6)])
def test_reject_overflow_before_device_transfer(state_length, option_length):
    def tokenizer(text, **kwargs):
        assert kwargs["truncation"] is False
        length = option_length if isinstance(text, list) else state_length
        return {"input_ids": SimpleNamespace(shape=(1, length))}

    with pytest.raises(ValueError, match="refusing truncation"):
        encode_inputs(tokenizer, "State", ["A", "B"], 16, 5)


@pytest.mark.parametrize("device,limit", [("cpu", None), ("cuda:0", 0), ("cuda:0", 16385)])
def test_invalid_settings_before_download(device, limit):
    with pytest.raises(ValueError):
        DMJEPAAdapter("unused", "main", device, context_limit=limit)
