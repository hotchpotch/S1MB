"""Descriptive diagnostics with baselines fixed by the entire benchmark.

Baselines use evaluation targets, not a separate training set. They measure
improvement over trivial predictions; they are not deployable fitted models.
"""

from statistics import median

from .data import Case, Prediction, Task

EPSILON = 1e-12


def diagnostics(
    cases: list[Case], predictions: list[Prediction], task: Task
) -> dict[str, float | None]:
    index = {
        (c.case_id, q.id): (q, c.targets[q.id].probabilities) for c in cases for q in c.questions
    }
    entries = list(index.values())
    rows = [
        (index[p.case_id, p.question_id], p.probabilities)
        for p in predictions
        if p.probabilities is not None
    ]
    n = len(rows)
    out: dict[str, float | None] = {}
    skill = None
    if task == "noul":
        prevalence = sum(t["true"] for _, t in entries) / len(entries)
        tp = sum(t["true"] for (_, t), p in rows if p["true"] >= 0.5)
        fp = sum(t["false"] for (_, t), p in rows if p["true"] >= 0.5)
        fn = sum(t["true"] for (_, t), p in rows if p["true"] < 0.5)
        tn = sum(t["false"] for (_, t), p in rows if p["true"] < 0.5)
        recall = tp / (tp + fn) if tp + fn > EPSILON else None
        specificity = tn / (tn + fp) if tn + fp > EPSILON else None
        balanced = (
            (recall + specificity) / 2 if (recall is not None and specificity is not None) else None
        )
        baseline = sum((t["true"] - prevalence) ** 2 for _, t in entries) / len(entries)
        brier = sum((p["true"] - t["true"]) ** 2 for (_, t), p in rows) / n if n else None
        skill = 2 * balanced - 1 if balanced is not None else None
        out = {
            "accuracy": (tp + tn) / n if n else None,
            "positive_recall": recall,
            "specificity": specificity,
            "precision": tp / (tp + fp) if tp + fp > EPSILON else None,
            "balanced_accuracy": balanced,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > EPSILON else None,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "positive_prevalence": prevalence,
            "majority_accuracy_baseline": max(prevalence, 1 - prevalence),
            "constant_brier_baseline": baseline,
            "brier_skill": 1 - brier / baseline
            if baseline > EPSILON and brier is not None
            else None,
        }
    elif task == "choice":
        uniform = sum(1 / len(q.options) for q, _ in entries) / len(entries)
        ids = {o.id for q, _ in entries for o in q.options}
        # An unavailable ID/position falls back to a uniform prediction.
        fixed_id = max(
            sum(t.get(key, 1 / len(q.options)) for q, t in entries) / len(entries) for key in ids
        )
        fixed_position = max(
            sum(
                t[q.options[i].id] if i < len(q.options) else 1 / len(q.options) for q, t in entries
            )
            / len(entries)
            for i in range(max(len(q.options) for q, _ in entries))
        )
        baseline = max(uniform, fixed_id, fixed_position)
        accuracy = (
            sum(t[max(q.options, key=lambda o: p[o.id]).id] for (q, t), p in rows) / n
            if n
            else None
        )
        skill = (
            (accuracy - baseline) / (1 - baseline)
            if (accuracy is not None and baseline < 1 - EPSILON)
            else None
        )
        out = {"uniform_accuracy_baseline": uniform, "fixed_answer_accuracy_baseline": baseline}
    else:

        def value(q, probabilities):
            scale = {o.id: float(o.value) for o in q.options}
            low, high = min(scale.values()), max(scale.values())
            return sum((v - low) / (high - low) * probabilities[k] for k, v in scale.items())

        targets = [value(q, t) for q, t in entries]
        constant = median(targets)
        baseline = sum(abs(t - constant) for t in targets) / len(targets)
        # Preserve the primary metric's arithmetic for approximately normalized inputs.
        errors = []
        for (q, t), p in rows:
            scale = {o.id: o.value for o in q.options if o.value is not None}
            errors.append(
                sum(v * (p[k] - t[k]) for k, v in scale.items())
                / (max(scale.values()) - min(scale.values()))
            )
        mae = sum(abs(e) for e in errors) / n if n else None
        skill = 1 - mae / baseline if baseline > EPSILON and mae is not None else None
        out = {
            "normalized_score_rmse": (sum(e * e for e in errors) / n) ** 0.5 if n else None,
            "constant_score_baseline": constant,
            "constant_mae_baseline": baseline,
        }
    out["baseline_adjusted_skill"] = skill
    out["baseline_adjusted_score"] = max(0.0, min(1.0, skill)) if skill is not None else None
    return out
