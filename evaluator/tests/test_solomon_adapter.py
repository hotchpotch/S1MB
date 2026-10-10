"""Solomon's gateway must preserve authored criteria and duplicate candidate positions."""

import json

from s1mb.adapters.sifr import sifr_questions
from s1mb.adapters.solomon import solomon_options


def test_structured_score_values_and_duplicate_positions():
    from test_upstream_adapters import cases

    case = cases()
    case.questions[2].options.reverse()
    wire = sifr_questions(case)
    values = [json.loads(value) for value in solomon_options(wire["rate"])]
    assert [value["value"] for value in values] == [30, 10]
    duplicate = {"criteria": {"option_0": "same", "option_1": "same", "option_2": "different"}}
    assert solomon_options(duplicate) == [
        "option_0: same", "option_1: same", "option_2: different",
    ]
    assert solomon_options({"criteria": {"option_0": "first", "option_1": "second"}}) == [
        "first", "second",
    ]
