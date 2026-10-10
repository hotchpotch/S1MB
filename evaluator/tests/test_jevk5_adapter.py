"""JevK5 native probability transport preserves numeric criteria."""
from types import SimpleNamespace

from s1mb.adapters.jevk5 import JevK5Adapter
from s1mb.data import InferenceCase


def test_structured_numeric_score_transport():
    def probabilities(state, question):
        assert question["type"] == "choice"
        assert [v["value"] for v in question["criteria"].values()] == [30.0, 10.0]
        assert question["criteria"]["option_0"]["description"] == {"urgency": "today"}
        return {"option_0": 0.7, "option_1": 0.3}, 10

    adapter = JevK5Adapter.__new__(JevK5Adapter)
    adapter.engine = SimpleNamespace(probabilities=probabilities)
    case = InferenceCase.model_validate({
        "case_id": "example", "state": "state",
        "questions": [{"id": "urgency", "task": "score", "instructions": "Rate",
                       "options": [{"id": "high", "description": "Urgent", "value": 30,
                                    "description_json": '{"urgency":"today"}'},
                                   {"id": "low", "description": "Routine", "value": 10}]}],
    })
    assert adapter.predict(case)[0].probabilities == {"high": 0.7, "low": 0.3}
