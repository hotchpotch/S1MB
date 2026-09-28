"""Deterministic demo adapter for offline integration tests."""

from s1mb.data import InferenceCase, ModelInfo, Prediction


class DummyAdapter:
    def predict(self, case: InferenceCase) -> list[Prediction]:
        return [
            Prediction(
                case_id=case.case_id,
                question_id=q.id,
                probabilities={o.id: 1 / len(q.options) for o in q.options},
            )
            for q in case.questions
        ]

    def metadata(self) -> ModelInfo:
        return ModelInfo(id="uniform-demo", adapter="dummy", revision="1")

    def close(self) -> None:
        pass
