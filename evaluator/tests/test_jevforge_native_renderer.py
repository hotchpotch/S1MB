import json
import unittest

from s1mb.adapters.base import questions_for_api
from s1mb.adapters.jevforge import native_questions
from s1mb.data import Option, Question


class NativeRendererTest(unittest.TestCase):
    def question(self, **overrides):
        args = {
            "id": "q",
            "task": "choice",
            "instructions": "pick",
            "options": [
                Option(id="node9", description="first"),
                Option(id="node2", description="second"),
            ],
        }
        args.update(overrides)
        return Question.model_validate(args)

    def test_plain_text_matches_existing_adapter(self):
        q = self.question(system_prompt="system")
        self.assertEqual(native_questions([q]), questions_for_api([q], anonymous_choice=True))

    def test_structured_instruction_restores_training_order_and_spacing(self):
        q = self.question(
            instructions='{"a":[1,"x"],"z":2}',
            instructions_json='{"z": 2, "a": [1, "x"]}',
            system_prompt="policy",
        )
        actual = native_questions([q])["q"]["instructions"]
        self.assertEqual(actual, 'policy\n\n{"z": 2, "a": [1, "x"]}')
        self.assertNotEqual(
            actual, questions_for_api([q], anonymous_choice=True)["q"]["instructions"]
        )

    def test_plain_json_string_stays_verbatim(self):
        text = 'Find the "quoted" value.\n中文'
        q = self.question(instructions=text, instructions_json=json.dumps(text))
        self.assertEqual(native_questions([q])["q"]["instructions"], text)

    def test_structured_criteria_keep_source_order(self):
        q = self.question(
            options=[
                Option(id="x", description='{"a":1,"z":2}', description_json='{"z":2,"a":1}'),
                Option(id="y", description="plain", description_json='"plain"'),
            ]
        )
        self.assertEqual(
            native_questions([q])["q"]["criteria"],
            {"option_0": '{"z": 2, "a": 1}', "option_1": "plain"},
        )

    def test_noul_order_and_score_levels_are_not_remapped(self):
        n = self.question(
            task="noul",
            options=[Option(id="true", description="yes"), Option(id="false", description="no")],
        )
        s = self.question(
            task="score",
            options=[
                Option(id="low", description="low", value=0),
                Option(id="high", description="high", value=0.5),
            ],
        )
        self.assertEqual(native_questions([n])["q"]["criteria"], {"true": "yes", "false": "no"})
        self.assertEqual(native_questions([s])["q"]["criteria"], ["low", "high"])


if __name__ == "__main__":
    unittest.main()
