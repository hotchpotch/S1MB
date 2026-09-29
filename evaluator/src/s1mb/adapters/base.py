"""Small model boundary; inference inputs never contain targets or provenance."""

import json
from typing import Protocol, cast

from s1mb.data import InferenceCase, ModelInfo, Prediction, Question, check_probabilities


class ModelAdapter(Protocol):
    def predict(self, case: InferenceCase) -> list[Prediction]: ...
    def metadata(self) -> ModelInfo: ...
    def close(self) -> None: ...


def questions_for_api(
    questions: list[Question],
    *,
    sort_score: bool = False,
    structured: bool = False,
    anonymous_choice: bool = False,
) -> dict:
    result = {}
    for q in questions:
        instructions = "\n\n".join(s for s in [q.system_prompt, q.instructions] if s)
        if structured:
            instructions = (
                json.loads(q.instructions_json) if q.instructions_json else q.instructions
            )
            if q.system_prompt:
                instructions = {"system": q.system_prompt, "instruction": instructions}

        def description(option):
            if structured and option.description_json is not None:
                return json.loads(option.description_json)
            return option.description

        criteria = (
            [description(o) for o in score_options(q, sort_score)]
            if q.task == "score"
            else {o.id: description(o) for o in q.options}
        )
        if anonymous_choice and q.task == "choice":
            criteria = {f"option_{i}": description(o) for i, o in enumerate(q.options)}
        result[q.id] = {"type": q.task, "instructions": instructions, "criteria": criteria}
    return result


def score_options(question: Question, ascending: bool):
    """Keep stored option order intact; sort only at the API boundary when requested."""
    return (
        sorted(question.options, key=lambda o: cast(float, o.value))
        if ascending
        else question.options
    )


def decode_answers(
    case: InferenceCase,
    answers: dict,
    *,
    rounded: bool = False,
    rounding_decimals: int = 4,
    sort_score: bool = False,
    anonymous_choice: bool = False,
) -> list[Prediction]:
    if set(answers) != {q.id for q in case.questions}:
        raise ValueError("Returned question IDs differ from the request")
    result = []
    for q in case.questions:
        answer = answers[q.id]
        if answer["type"] != q.task:
            raise ValueError("Returned task differs from the request")
        if q.task == "noul":
            value = float(answer["noul"])
            probabilities = {"false": 1 - value, "true": value}
        elif q.task == "score":
            raw = answer["probabilities"]
            if set(raw) != {str(i) for i in range(len(q.options))}:
                raise ValueError("Score level IDs do not match")
            probabilities = {
                o.id: float(raw[str(i)]) for i, o in enumerate(score_options(q, sort_score))
            }
        else:
            probabilities = {k: float(v) for k, v in answer["probabilities"].items()}
            if anonymous_choice:
                expected = {f"option_{i}" for i in range(len(q.options))}
                if set(probabilities) != expected:
                    raise ValueError("Anonymous choice keys do not match")
                probabilities = {
                    o.id: probabilities[f"option_{i}"] for i, o in enumerate(q.options)
                }
        if rounded:
            # Normalize only within the configured decimal rounding error.
            total = sum(probabilities.values())
            tolerance = 0.5 * 10 ** (-rounding_decimals) + 1e-6
            if not 0 < total or abs(total - 1) > len(probabilities) * tolerance:
                raise ValueError("Probability sum exceeds the allowed rounding error")
            if any(not 0 <= p <= 1 for p in probabilities.values()):
                raise ValueError("Invalid probability")
            probabilities = {k: v / total for k, v in probabilities.items()}
        check_probabilities(probabilities, [o.id for o in q.options])
        result.append(
            Prediction(case_id=case.case_id, question_id=q.id, probabilities=probabilities)
        )
    return result
