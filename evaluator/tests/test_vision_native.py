"""Visual transports retain structured instructions, truth definitions and numeric scales."""

import json

from test_upstream_adapters import cases

from s1mb.adapters.vision_native import choice_transport, typed_questions


def test_structured_instruction_remains_complete_text():
    case = cases()
    case.questions[0].instructions_json = json.dumps({
        "instruction": "Select an image.", "rules": {"count": 16},
    })
    questions = typed_questions(case)
    assert json.loads(questions["choose"]["instructions"]) == {"system": "System.", "instruction": {
        "instruction": "Select an image.", "rules": {"count": 16},
    }}
    assert "gold-must-not-be-sent" not in str(questions)


def test_binary_definition_alignment_and_numeric_score_anchors():
    questions = typed_questions(cases())
    assert choice_transport(questions["judge"]) == ["No", "Yes"]
    values = [json.loads(v) for v in choice_transport(questions["rate"])]
    assert values == [{"value": 10.0, "description": "Low"},
                      {"value": 30.0, "description": "High"}]
    assert choice_transport(questions["choose"]) == ["Same", "Same"]
