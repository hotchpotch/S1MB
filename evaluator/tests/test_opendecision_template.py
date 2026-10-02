"""Authored JSON and braces must survive native hypothesis formatting."""

import pytest

from s1mb.adapters.opendecision import LiteralHypothesisClassifier


@pytest.mark.parametrize(
    "instruction", ["Pick one", '{"element_id": 3}', "Use {} and {name}", "Unmatched { or }"]
)
def test_hypothesis_preserves_authored_instruction(instruction):
    def classifier(sequence, **kwargs):
        return kwargs["hypothesis_template"].format('{"candidate": 4}')

    wrapped = LiteralHypothesisClassifier(classifier)
    template = f'The answer to the question "{instruction}" is {{}}.'
    assert (
        wrapped("state", hypothesis_template=template)
        == f'The answer to the question "{instruction}" is {{"candidate": 4}}.'
    )


def test_default_template_is_unchanged():
    def classifier(sequence, **kwargs):
        return sequence, kwargs

    assert LiteralHypothesisClassifier(classifier)("state", candidate_labels=["a"]) == (
        "state",
        {"candidate_labels": ["a"]},
    )
