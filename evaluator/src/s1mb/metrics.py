"""Task metrics preserve soft targets and original Score scales."""

from .data import METRICS, Box, Case, Prediction, Task
from .diagnostics import diagnostics


def calculate(
    cases: list[Case], predictions: list[Prediction], task: Task
) -> dict[str, float | None]:
    index = {(c.case_id, q.id): (q, c.targets[q.id]) for c in cases for q in c.questions}
    if task == "box":
        ious = []
        for pred in predictions:
            if pred.abstention is not None:
                ious.append(0.0)
            elif pred.box is not None:
                target_box = index[pred.case_id, pred.question_id][1].box
                assert target_box is not None
                ious.append(box_iou(pred.box, target_box))
        return {
            "mean_iou": sum(ious) / len(ious) if ious else None,
            "accuracy_at_iou_0_5": sum(v >= 0.5 for v in ious) / len(ious) if ious else None,
        }
    values = []
    for pred in predictions:
        if pred.abstention is not None:
            values.append(0.0 if task == "choice" else 1.0)
            continue
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


def box_iou(a: Box, b: Box) -> float:
    intersection = max(0.0, min(a.right, b.right) - max(a.left, b.left)) * max(
        0.0, min(a.bottom, b.bottom) - max(a.top, b.top)
    )
    union = (
        (a.right - a.left) * (a.bottom - a.top)
        + (b.right - b.left) * (b.bottom - b.top)
        - intersection
    )
    return intersection / union if union > 0 else 0.0
