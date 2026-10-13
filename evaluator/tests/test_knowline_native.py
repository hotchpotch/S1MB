"""Native KnowLine transport preserves candidate identities and authored scales."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.knowline_native import KnowLineNativeAdapter


def test_native_decoding_retains_authored_order_and_numeric_scale():
    case = cases()
    case.questions[2].options.reverse()
    adapter = KnowLineNativeAdapter.__new__(KnowLineNativeAdapter)

    def run(state, questions):
        assert state == case.state
        assert set(questions) == {"decision"}
        question = questions["decision"]
        task = question["type"]
        if task == "noul":
            assert question["criteria"] == {"true": "Yes", "false": "No"}
            answer = {"type": task, "noul": 0.7}
        else:
            if task == "choice":
                assert list(question["criteria"]) == ["option_0", "option_1"]
                keys = list(question["criteria"])
            else:
                assert [item["value"] for item in question["criteria"]] == [30, 10]
                keys = ["0", "1"]
            answer = {"type": task, "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}
        assert "gold-must-not-be-sent" not in str(questions)
        return {"decision": answer}, {}

    adapter.engine = SimpleNamespace(run=run)
    result = adapter.predict(case)
    assert result[0].probabilities == {"x": 0.2, "y": 0.8}
    assert result[1].probabilities["true"] == 0.7
    assert result[2].probabilities == {"high": 0.2, "low": 0.8}


def test_missing_native_answer_is_rejected():
    adapter = KnowLineNativeAdapter.__new__(KnowLineNativeAdapter)
    adapter.engine = SimpleNamespace(run=lambda *args: ({}, {}))
    with pytest.raises(ValueError, match="answer count"):
        adapter.predict(cases())
