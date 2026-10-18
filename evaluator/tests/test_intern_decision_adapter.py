"""Native masked-symbol typed mapping and runtime requirements."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_upstream_adapters import cases

from s1mb.adapters import intern_decision
from s1mb.adapters.intern_decision import InternDecisionAdapter, intern_questions


def test_native_criteria_preserve_structured_authored_values_and_order():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested":true}'
    case.questions[2].options.reverse()
    questions = intern_questions(case)
    assert questions["choose"]["criteria"] == {"option_0": {"nested": True}, "option_1": "Same"}
    assert questions["judge"]["criteria"] == {"true": "Yes", "false": "No"}
    assert questions["rate"]["criteria"] == {"30.0": "High", "10.0": "Low"}
    assert "gold-must-not-be-sent" not in str(questions)


def test_native_field_alignment_without_case_or_question_identifiers():
    adapter = InternDecisionAdapter.__new__(InternDecisionAdapter)
    adapter.image_receipts = {}

    def predict(request):
        assert set(request) == {"state", "questions"}
        assert request["state"] == cases().state
        assert set(request["questions"]) == {"decision"}
        question = request["questions"]["decision"]
        task = question["type"]
        if task == "noul":
            answer = {"type": task, "noul": 0.7}
        else:
            keys = list(question["criteria"]) if task == "choice" else ["10.0", "30.0"]
            answer = {"type": task, "probabilities": dict(zip(keys, [0.2, 0.8], strict=True))}
        return {"answers": {"decision": answer}}

    adapter.engine = SimpleNamespace(predict=predict)
    result = adapter.predict(cases())
    assert result[0].probabilities == {"x": 0.2, "y": 0.8}
    assert result[1].probabilities["true"] == 0.7
    assert result[2].probabilities == {"low": 0.2, "high": 0.8}


def test_misaligned_native_fields_rejected():
    adapter = InternDecisionAdapter.__new__(InternDecisionAdapter)
    adapter.image_receipts = {}
    adapter.engine = SimpleNamespace(predict=lambda request: {"answers": {"other": {}}})
    with pytest.raises(ValueError, match="count"):
        adapter.predict(cases())


def test_invalid_budget_before_model_load():
    with pytest.raises(ValueError, match="positive"):
        InternDecisionAdapter("unused", "main", "unused", "cuda:0", context_limit=0)


def test_native_python_requirement(monkeypatch):
    monkeypatch.setattr(intern_decision.sys, "version_info", (3, 11, 0, "final", 0))
    with pytest.raises(RuntimeError, match="Python 3.12"):
        InternDecisionAdapter("unused", "main", "unused", "cuda:0")


def test_ordered_images_use_neutral_paths_without_exposing_them_in_state(monkeypatch):
    from PIL import Image

    case = cases()
    images = [Image.new("RGB", (2, 3), color) for color in ["red", "blue"]]
    monkeypatch.setattr(intern_decision, "ordered_images", lambda _: (images, case.state, [1, 2]))
    adapter = InternDecisionAdapter.__new__(InternDecisionAdapter)
    adapter.image_receipts = {}
    paths = []

    def predict(request):
        assert request["state"] == case.state
        assert "s1mb-intern-images-" not in str(request["state"])
        assert [Path(p).name for p in request["images"]] == ["image-1.png", "image-2.png"]
        for source, expected in zip(request["images"], images, strict=True):
            with Image.open(source) as actual:
                assert actual.tobytes() == expected.tobytes()
        paths.extend(request["images"])
        return {"answers": {"decision": {"type": "noul", "noul": 0.7}}}

    case.questions = [case.questions[1]]
    adapter.engine = SimpleNamespace(predict=predict)
    assert adapter.predict(case)[0].probabilities["true"] == 0.7
    assert all(not Path(path).exists() for path in paths)
