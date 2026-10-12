"""Torchcast must retain task calibration and restore mutable native state on failure."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.torchcast import TorchcastAdapter


@pytest.mark.parametrize("fail", [False, True])
def test_task_calibration_order_and_restoration(fail):
    from test_upstream_adapters import cases

    case = cases()
    case.questions[1].options.reverse()
    case.questions[2].options.reverse()
    temperatures = {"choice": 1.2, "noul": 1.4, "score": 1.3}
    seen = []

    def decide(state, questions):
        key, spec = next(iter(questions.items()))
        seen.append((key, temperatures["choice"], spec))
        if fail:
            raise ValueError("native failure")
        keys = list(spec["criteria"])
        return {key: {"probabilities": dict.fromkeys(keys, 1 / len(keys))}}, {}

    adapter = TorchcastAdapter.__new__(TorchcastAdapter)
    adapter.engine = SimpleNamespace(temperature=temperatures, decide=decide)
    if fail:
        with pytest.raises(ValueError, match="native failure"):
            adapter.predict(case)
    else:
        predictions = adapter.predict(case)
        assert [entry[1] for entry in seen] == [1.2, 1.4, 1.3]
        assert list(seen[1][2]["criteria"]) == ["false", "true"]
        assert [value["value"] for value in seen[2][2]["criteria"].values()] == [30, 10]
        assert list(predictions[1].probabilities) == ["false", "true"]
    assert temperatures == {"choice": 1.2, "noul": 1.4, "score": 1.3}
