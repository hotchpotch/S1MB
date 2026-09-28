# Scoring and baseline-adjusted-v1

Original primary metrics and benchmark definitions are unchanged. Additional
metrics describe different aspects of the saved predictions. The experimental
adjusted index is a convenience for comparison, not a validated measure of
universal capability. Always inspect the task and benchmark results beside it.

## Noul

- **Brier** (primary, lower is better): mean `(p(true) - target(true))²`.
- Predict true when `p(true) >= 0.5`, including exact ties. This threshold is fixed
  and is not tuned on test results.
- **Accuracy**: `(TP + TN) / N`.
- **Precision**: `TP / (TP + FP)`.
- **True Recall**: `TP / (TP + FN)`.
- **False Recall / Specificity**: `TN / (TN + FP)`.
- **F1**: `2 TP / (2 TP + FP + FN)`.
- **Balanced Accuracy (BA)**: `(True Recall + False Recall) / 2`.
- **Adjusted skill**: `2 BA - 1`. Always-true and always-false predictions both
  score zero when both target classes are present.
- **Brier skill**: `1 - Brier / constant Brier`. The constant predicts the entire
  benchmark's mean target probability. This is a separate diagnostic; the
  combined index uses adjusted BA, not Brier skill.

Soft targets are never rounded. For a predicted true, target probability `t`
contributes `t` to TP and `1-t` to FP; predicted false contributes `t` to FN and
`1-t` to TN. Confusion counts may therefore be fractional. Accuracy and BA are
expected label agreement; Brier evaluates agreement with the probability target.
Even predicting the target probability exactly need not yield BA = 1 for soft
labels. Zero-denominator metrics are null, not zero or perfect.

Example: TP = 1, FN = 10, FP = 0, TN = 89 gives Accuracy = 90%, true Recall =
9.09%, BA = 54.55%, and adjusted skill = 9.09%. Always false gives Accuracy =
89%, BA = 50%, and adjusted skill = 0. These are classification measures, not
ranking nDCG.

The binary adjusted BA formula follows the chance-adjustment described in
[scikit-learn's balanced accuracy documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).

## Choice

The primary metric remains target probability at the selected option (Accuracy
for hard labels). Argmax ties use the original candidate order.

The baseline `b` is the best entire-benchmark expected accuracy among:

1. Uniform random selection, accounting for each question's candidate count.
2. Always selecting a particular option ID.
3. Always selecting a particular candidate position (original order).

A fixed ID or position missing from a question falls back to uniform selection.
Using both IDs and positions detects obvious answer-label/order artifacts even
when IDs and candidate ordering differ across questions. It does not establish
that IDs have consistent semantic meaning across examples; it only defines a
trivial prediction policy. We do not compute macro-F1 across unrelated option IDs.

Adjusted skill is `(accuracy - b) / (1 - b)`. The reference ceiling is 1; soft
label agreement may have a lower attainable ceiling. A baseline of 1 leaves no
headroom: adjustment is undefined and this benchmark is excluded for all models.
The current balanced release repairs the earlier CrowS-Pairs fixed-answer
artifact; its best fixed-answer baseline is now 52%. Raw results remain visible.

## Score

Normalize each option's value by `(value - min) / (max - min)` and calculate
predicted and target expected values. Report **MAE** (primary) and **RMSE** in
these normalized units. They measure expected-value error, not whether the full
predicted probability distribution matches the target distribution.

The constant baseline predicts the median of all normalized target expected
values, an absolute-error-minimizing constant. Baseline MAE is the mean absolute
error of this constant. Adjusted skill is `1 - MAE / baseline MAE`. If baseline
MAE is zero, adjustment is undefined.

## Aggregation and eligibility

- First average decisions within each benchmark for the underlying metric.
- Clip adjusted skill to `[0, 1]` **per benchmark** to produce the adjusted score.
  Zero means at or below the trivial baseline; one means the reference ceiling.
  Unclipped skills remain available, including negative values. Clipping avoids
  an arbitrarily negative error ratio dominating an index but hides how far below
  baseline a model performs, which is why raw metrics and skills must remain visible.
- The adjusted task score is the equally weighted mean of eligible benchmarks.
- The experimental overall index is the equally weighted mean of the three task
  scores. The current category has 60 Choice, 59 Noul, and 21 Score benchmarks;
  each task still receives one third of the index. This is an explicit convention, not an empirically validated weight.
- Eligibility is determined from the entire fixed dataset, identically for every
  model: both classes must have positive target mass for Noul; Choice needs
  baseline < 1; Score needs baseline MAE > 0. Numerical tolerance is `1e-12`.
- All required benchmarks must have complete measured results before ranking,
  **including** benchmarks excluded from adjustment. Missing, failed, or demo
  results cannot improve a rank by reducing the denominator. A task with no
  eligible benchmarks, or an undefined eligible metric, has no adjusted score.
- Diagnostic column averages use equal benchmark weights and show N/A if any
  required value is undefined; undefined values are not silently dropped.

Baselines are calculated using the **evaluation targets**, without model input
or training. They are descriptive best-constant references, not estimates of
held-out performance of a baseline fitted on training data. Their optimism and
sampling noise matter, particularly for small/imbalanced samples. No confidence
intervals are provided in v1; small differences should not be treated as decisive.
A majority-class correction does not fix mislabeled examples, answer artifacts,
input truncation, inappropriate prompts, or missing information. Existing dataset
and adapter audits still apply.

## Storage and validation

Python evaluations store primary and additional metrics. Validation requires the
full metric set and recomputes every value. The viewer reads the same HF Arrow
shards server-side and sends only summaries to the browser. It does not invoke
a model. Shared fixtures exercise imbalance, soft targets, variable option
counts, degenerate baselines, and scaled scores. Archived results are not loaded
into the new category.

The general-purpose columns show the equal-weight mean of the diverse and
contextual benchmarks for each task. These benchmarks also contribute to the
full task scores; the three general-purpose columns add no separate overall
weight. Both subsets must be complete for a general-purpose task score.
