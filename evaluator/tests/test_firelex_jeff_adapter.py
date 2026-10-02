"""Firelex Jeff fidelity tests without upstream dependencies or model weights."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.firelex_jeff import FirelexJeffAdapter, decision_row


def test_structured_values_ids_and_numeric_score_order():
    case = cases()
    q = case.questions[0]
    q.instructions_json = '{"rubric": [1, 2]}'
    q.options[0].description_json = '{"nested": true}'
    row = decision_row(case.state, q)
    assert row["state"] == case.state
    assert row["question"]["instructions"] == {
        "system": "System.",
        "instruction": {"rubric": [1, 2]},
    }
    assert row["question"]["criteria"] == {
        "option_0": {"nested": True},
        "option_1": "Same",
    }
    assert "gold-must-not-be-sent" not in str(row)
    q = case.questions[2]
    q.options.reverse()
    assert decision_row(case.state, q)["question"]["criteria"] == [
        {"value": 30.0, "description": "High"},
        {"value": 10.0, "description": "Low"},
    ]
    assert decision_row(case.state, case.questions[1])["question"]["criteria"] == {
        "true": "Yes",
        "false": "No",
    }


class FakeLogits:
    def __init__(self, values):
        self.values = values

    def __truediv__(self, temperature):
        assert temperature == 1.5
        return self

    def softmax(self, dim):
        assert dim == -1
        return self

    def __getitem__(self, key):
        assert key == (0, slice(None, 2))
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeEngine:
    temperature = 1.5

    def __init__(self, values):
        self.values = values
        self.rows = []

    def prepare(self, rows, max_length):
        assert max_length == 8192
        self.rows.extend(rows)
        return rows

    def __call__(self, batch):
        return FakeLogits(self.values)


def fake_adapter(values=(0.3, 0.7)):
    adapter = FirelexJeffAdapter.__new__(FirelexJeffAdapter)
    adapter.max_candidates = 26
    adapter.context_limit = 8192
    adapter.torch = cast(Any, SimpleNamespace(inference_mode=nullcontext))
    adapter.engine = FakeEngine(list(values))
    return adapter


def test_native_noul_order_maps_back_and_calibration_is_applied():
    adapter = fake_adapter()
    result = adapter.predict(cases())
    assert result[0].probabilities == {"x": 0.3, "y": 0.7}
    assert result[1].probabilities == {"false": 0.3, "true": 0.7}
    assert result[2].probabilities == {"low": 0.3, "high": 0.7}
    assert len(adapter.engine.rows) == 3


@pytest.mark.parametrize("values", [(float("nan"), 0.7), (-0.1, 1.1), (0.2, 0.3), (1.0,)])
def test_invalid_native_probabilities_are_not_normalized(values):
    with pytest.raises(ValueError):
        fake_adapter(values).predict(cases())


def test_candidate_and_context_overflow_do_not_run_inference():
    adapter = fake_adapter()
    adapter.max_candidates = 1
    with pytest.raises(ValueError, match="refusing to drop"):
        adapter.predict(cases())
    assert not adapter.engine.rows
    adapter.max_candidates = 26

    def overflow(*args, **kwargs):
        raise ValueError("Question branch exceeds the 8192-token limit; no input was truncated.")

    adapter.engine.prepare = overflow
    with pytest.raises(ValueError, match="no input was truncated"):
        adapter.predict(cases())
