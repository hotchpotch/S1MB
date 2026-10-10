"""Lossless input rejection and typed-label alignment without private artifacts."""

from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters.jevk5_lite import JevK5LiteAdapter, lite_inputs


def test_authored_definitions_numeric_levels_and_identifiers():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": true}'
    case.questions[2].options.reverse()
    rendered = [lite_inputs(case.state, q) for q in case.questions]
    assert rendered[0][2] == ['option 0: {"nested": true}', "option 1: Same"]
    assert rendered[1][2] == ["option 0: Yes", "option 1: No"]
    assert rendered[2][2] == ["level 0: 30.0: High", "level 1: 10.0: Low"]
    assert "gold-must-not-be-sent" not in str(rendered)


def adapter(encoded_length=5, body_length=2):
    model = JevK5LiteAdapter.__new__(JevK5LiteAdapter)
    model.context_limit = 5
    model.native = SimpleNamespace(_heads=lambda tasks: tasks)
    model.engine = SimpleNamespace(
        encode=lambda text, heads: (list(range(encoded_length)), [], [], (1, 1 + body_length)),
        _piece=lambda text: [1, 2],
        classify=lambda text, tasks: {
            task: {"probabilities": dict(zip(labels, [0.2, 0.8], strict=True))}
            for task, labels in tasks.items()
        },
    )
    return model


def test_alignment_for_all_tasks_and_duplicate_definitions():
    results = adapter().predict(cases())
    assert results[0].probabilities == {"x": 0.2, "y": 0.8}
    assert results[1].probabilities == {"true": 0.2, "false": 0.8}
    assert results[2].probabilities == {"low": 0.2, "high": 0.8}


@pytest.mark.parametrize("length,body", [(6, 2), (5, 1)])
def test_schema_overflow_and_native_body_truncation_reject_before_inference(length, body):
    model = adapter(length, body)

    def forbidden(*args):
        raise AssertionError("Overflow reached inference")

    model.engine.classify = forbidden
    with pytest.raises(ValueError, match="refusing truncation"):
        model.predict(cases())


def test_invalid_distribution_rejected():
    model = adapter()
    model.engine.classify = lambda text, tasks: {
        task: {"probabilities": dict.fromkeys(labels, float("nan"))}
        for task, labels in tasks.items()
    }
    with pytest.raises(ValueError):
        model.predict(cases())


@pytest.mark.parametrize("limit", [0, 513])
def test_invalid_limit_before_download(limit):
    with pytest.raises(ValueError):
        JevK5LiteAdapter("unused", "main", "unused", "cuda:0", context_limit=limit)
