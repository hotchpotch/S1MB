"""Task metrics preserve soft targets and original Score scales."""

from .data import METRICS, Case, Prediction, Task
from .diagnostics import diagnostics


def calculate(
    cases: list[Case], predictions: list[Prediction], task: Task
) -> dict[str, float | None]:
    index = {(c.case_id, q.id): (q, c.targets[q.id]) for c in cases for q in c.questions}
    values = []
    for pred in predictions:
        if pred.probabilities is None:
            continue
        q, target = index[pred.case_id, pred.question_id]
        p, t = pred.probabilities, target.probabilities
        if task == "choice":
            # Resolve exact ties in the original candidate order.
            chosen = max(q.options, key=lambda o: p[o.id]).id
            value = t[chosen]
        elif task == "noul":
            value = (p["true"] - t["true"]) ** 2
        else:
            scale = {o.id: float(o.value) for o in q.options if o.value is not None}
            value = abs(sum(scale[k] * (p[k] - t[k]) for k in scale)) / (
                max(scale.values()) - min(scale.values())
            )
        values.append(value)
    return {
        METRICS[task]: sum(values) / len(values) if values else None,
        **diagnostics(cases, predictions, task),
    }
