"""Native Nimble v3 tuple responses retain authored numeric score levels."""
from s1mb.adapters.nimble_v3 import NimbleV3Adapter
from s1mb.data import InferenceCase


def test_native_response_and_numeric_values():
    def engine(state, questions):
        spec = questions["decision"]
        assert spec["type"] == "choice"
        assert [v["value"] for v in spec["criteria"].values()] == [30.0, 10.0]
        return ({"answers": {"decision": {"probabilities": {
            "option_0": 0.7, "option_1": 0.3,
        }}}}, None)

    adapter = NimbleV3Adapter.__new__(NimbleV3Adapter)
    adapter.engine = engine
    case = InferenceCase.model_validate({
        "case_id": "example", "state": "state",
        "questions": [{"id": "urgency", "task": "score", "instructions": "Rate urgency",
                       "options": [{"id": "high", "description": "Urgent", "value": 30},
                                   {"id": "low", "description": "Routine", "value": 10}]}],
    })
    assert adapter.predict(case)[0].probabilities == {"high": 0.7, "low": 0.3}
