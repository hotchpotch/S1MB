"""Verify native head alignment and rejection of silently shortened inputs."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.saihai_310m import Saihai310mAdapter


def adapter(shortened=False):
    instance = Saihai310mAdapter.__new__(Saihai310mAdapter)
    group = SimpleNamespace(query="query", query_parts=None, candidates=["a", "b"], task="choice")
    instance.native = SimpleNamespace(input_groups=lambda request, **kwargs: [group])
    instance.runtime = SimpleNamespace(
        prefix_layout="instruction_state",
        prepare_groups=lambda groups: [SimpleNamespace(
            query=[1] if shortened else [1, 2], documents=[[3], [4]],
        )],
        _queries=lambda *args: [[1, 2]],
        _documents=lambda *args: [[3], [4]],
    )
    def predict(request, **kwargs):
        assert [d["id"] for d in request["decisions"]] == [
            "question_0", "question_1", "question_2",
        ]
        return {
            d["id"]: {"probabilities": dict(zip(
                [c["id"] for c in d["criteria"]], [0.2, 0.8], strict=True,
            ))}
            for d in request["decisions"]
        }
    instance.engine = SimpleNamespace(predict=predict)
    return instance


def test_anonymous_slots_restore_authored_ids_and_score_order():
    from test_upstream_adapters import cases

    predictions = adapter().predict(cases())
    assert predictions[0].probabilities == {"x": 0.2, "y": 0.8}
    assert predictions[1].probabilities == {"true": 0.2, "false": 0.8}
    assert predictions[2].probabilities == {"low": 0.2, "high": 0.8}


def test_native_shortening_is_rejected_before_inference():
    from test_upstream_adapters import cases

    with pytest.raises(ValueError, match="refusing truncation"):
        adapter(shortened=True).predict(cases())


def test_explicit_native_adaptive_budget_accepts_native_shortening():
    from test_upstream_adapters import cases

    instance = adapter(shortened=True)
    instance.native_adaptive_budget = True
    assert len(instance.predict(cases())) == 3
