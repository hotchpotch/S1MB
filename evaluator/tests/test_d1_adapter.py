"""d1 boundary checks without checkpoint downloads or GPU inference."""

import io
from typing import Any, cast

import pytest
from PIL import Image
from test_upstream_adapters import cases

from s1mb.adapters.d1 import D1Adapter, d1_questions, decode_answer
from s1mb.data import ImageAsset, InferenceCase, Question


def test_questions_preserve_semantics_and_anonymize_ids():
    case = cases()
    case.questions[0].options[0].description_json = '{"nested": "日本語"}'
    case.questions[2].options.reverse()
    choice, noul, score = d1_questions(case).values()
    assert choice["instructions"] == "System.\n\nChoose."
    assert choice["criteria"] == {"option_0": '{"nested": "日本語"}', "option_1": "Same"}
    assert noul["criteria"] == {"true": "Yes", "false": "No"}
    assert score["criteria"] == ["Low", "High"]
    assert "gold-must-not-be-sent" not in str([choice, noul, score])
    assert "boundary" not in str([choice, noul, score])


def test_answers_map_to_declared_options():
    case = cases()
    case.questions[2].options.reverse()
    choice, noul, score = case.questions
    assert decode_answer(
        case, choice, {"probabilities": {"option_0": 0.2, "option_1": 0.8}}, []
    ).probabilities == {"x": 0.2, "y": 0.8}
    assert decode_answer(case, noul, {"noul": 0.75}, []).probabilities == {
        "true": 0.75,
        "false": 0.25,
    }
    assert decode_answer(
        case, score, {"probabilities": {"0": 0.3, "1": 0.7}}, []
    ).probabilities == {"low": 0.3, "high": 0.7}
    with pytest.raises(ValueError, match="Choice options"):
        decode_answer(case, choice, {"probabilities": {"x": 0.2, "y": 0.8}}, [])
    with pytest.raises(ValueError, match="Score levels"):
        decode_answer(case, score, {"probabilities": {"0": 1.0}}, [])


def box_case(tmp_path, images=1) -> InferenceCase:
    assets = {}
    for index in range(images):
        path = tmp_path / f"image_{index + 1}.png"
        buffer = io.BytesIO()
        Image.new("RGB", (300, 300), (index, 0, 0)).save(buffer, format="PNG")
        path.write_bytes(buffer.getvalue())
        import hashlib

        assets[f"image_{index + 1}"] = ImageAsset(
            path=str(path), sha256=hashlib.sha256(buffer.getvalue()).hexdigest(), policy="scene-896"
        )
    return InferenceCase(
        case_id="boundary",
        state={"images": list(assets)},
        images=assets,
        questions=[Question(id="where", task="box", instructions="Where?")],
    )


def test_box_requires_one_image_and_inverts_padding(tmp_path):
    case = box_case(tmp_path)
    assert d1_questions(case) == {"field_0": {"type": "box", "instructions": "Where?"}}
    with pytest.raises(ValueError, match="exactly one image"):
        d1_questions(box_case(tmp_path, images=2))
    receipt = {"payload_size": [200, 100], "content_rectangle": [50, 0, 100, 100]}
    edges = {"left": 0.25, "right": 0.5, "top": 0.1, "bottom": 0.9}
    prediction = decode_answer(case, case.questions[0], {"box": edges}, [receipt])
    assert prediction.box is not None
    assert prediction.box.left == pytest.approx(0)
    assert prediction.box.right == pytest.approx(0.5)
    assert prediction.box.top == pytest.approx(0.1)


def test_documented_missing_box_is_an_abstention(tmp_path):
    case = box_case(tmp_path)
    prediction = decode_answer(case, case.questions[0], {"box": None}, [])
    assert prediction.abstention == "cannot_determine"
    assert prediction.box is None


class FakeNative:
    def __init__(self, length):
        self.length = length

    def named(self, questions):
        return questions

    def _sequences(self, requests):
        return [(0, 0, list(range(self.length)))]


class FakeModel:
    def __init__(self):
        self.calls = []

    def system_one(self, state, questions, media):
        self.calls.append((state, questions, media))
        return {
            "answers": {
                "field_0": {"probabilities": {"option_0": 0.2, "option_1": 0.8}},
                "field_1": {"noul": 0.75},
                "field_2": {"probabilities": {"0": 0.3, "1": 0.7}},
            }
        }


def adapter(length=10, limit=10):
    model = D1Adapter.__new__(D1Adapter)
    model.system_one = cast(Any, FakeNative(length))
    model.engine = FakeModel()
    model.context_limit = limit
    model.image_receipts = {}
    return model


def test_predict_passes_authored_state_and_rejects_overflow():
    model = adapter()
    predictions = model.predict(cases())
    assert model.engine.calls[0][0] == {"evidence": "text"}
    assert model.engine.calls[0][2] is None
    assert [p.question_id for p in predictions] == ["choose", "judge", "rate"]
    overflowing = adapter(length=11)
    with pytest.raises(ValueError, match="refusing truncation"):
        overflowing.predict(cases())
    assert overflowing.engine.calls == []


@pytest.mark.parametrize("limit", [0, 128_001])
def test_context_limit_cannot_exceed_native_capacity(limit):
    with pytest.raises(ValueError, match="context_limit"):
        D1Adapter("model", "revision", "cuda", context_limit=limit)
