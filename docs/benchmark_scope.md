# Benchmark scope and interpretation

S1MB (System One Mosaic Benchmark) combines specialized datasets through three
typed decision interfaces. It compares saved model predictions under recorded
evaluation conditions. It does not establish general intelligence, unseen-task
generalization, or training-data non-overlap.

## Decision types

| Task | Question contract | Prediction | Primary metric |
| --- | --- | --- | --- |
| Choice | Choose among authored alternatives | A probability distribution over those alternatives | Target probability at the selected alternative; higher is better |
| Noul | Judge the authored true/false condition | Probabilities for both categories | Brier score against the true-category target; lower is better |
| Score | Assess input against criteria with numeric levels | A probability distribution over those levels | Normalized expected-value MAE; lower is better |

An input can contain structured values and multiple questions. Adapters preserve
criterion order, dataset-default instructions, Noul definitions, numeric Score
levels, and soft targets. A ranking distribution is not a numeric Score label.
Targets, provenance, and identifiers stay out of model text. See the
[adapter contract](adapters.md) for implementation details.

## Active coverage

The tracked [English category](../evaluator/data/categories/english-v1.json)
currently contains 137 benchmarks from 106 dataset subsets:

| Task | Benchmarks |
| --- | ---: |
| Choice | 57 |
| Noul | 59 |
| Score | 21 |

A subset can support more than one decision type, so subset and benchmark counts
differ. Only the active evaluation manifest controls dataset membership; tracked
benchmark/category definitions must agree with it. Downloaded or quarantined
files do not automatically become evaluation members.

The [smoke category](../evaluator/data/categories/smoke-v1.json) selects one
benchmark per task for integration checks. It is not a representative performance
sample. After acquiring the dataset, inspect local coverage from `evaluator/`:

```sh
uv run s1mb list --category english-v1
uv run s1mb check-data --category english-v1
```

### Generalization subset

The active category includes six benchmarks: Diverse and Contextual variants for
each of Choice, Noul, and Score. `--generalization-only` restricts evaluation to
this subset and intersects with other selection filters. The viewer's
**Generalization tasks only** control recomputes coverage and rankings within it.
A model complete in this subset can participate there without full-category
coverage. The subset's name does not establish unseen-task generalization or
training-data non-overlap.

## Reading the leaderboard

**Task Avg** is a baseline-adjusted index on a 0–100 scale. Scores are adjusted
and clipped per benchmark, averaged within each task, and then averaged across
the three equally weighted tasks. Zero means at or below the reference baseline;
100 is a reference ceiling. Noul uses balanced accuracy for this adjustment,
while its primary raw metric remains Brier.

**Borda Score**, the default sorting column, averages relative rank points across
all active benchmarks with equal benchmark weights. It depends on the complete
model roster and does not give each task equal weight. Adding or removing models
can change Borda scores without changing any predictions.

Missing aggregates are unavailable, not zero. Complete measured coverage is
required in the selected scope; dummy predictions cannot qualify. Detail views
retain raw metric directions. Neither `1 - MAE` nor Borda is accuracy. Consult the
[scoring specification](../evaluator/SCORING.md) for formulas, ties, baseline
eligibility, and ranking requirements.

Baselines are descriptive references computed from evaluation targets, not
estimates of held-out baseline performance. The adjusted index is experimental;
no confidence intervals are supplied. Inspect per-task and per-benchmark results
and evaluation conditions before treating small differences as meaningful.

## Reproducibility and data boundaries

The [evaluation dataset](https://huggingface.co/datasets/hotchpotch/s1mb-dataset)
holds inputs and targets. Each result records its dataset repository and exact
commit SHA, model revision, and evaluator provenance. Published model folders may
combine results from different recorded releases; each is validated and its
baselines computed against its own release. Current definitions determine
leaderboard membership. See [evaluation](evaluation.md#dataset-revisions) and
[result submission](contributing_results.md).

The [results dataset](https://huggingface.co/datasets/hotchpotch/s1mb-result)
holds published measurements and display metadata. The viewer reads saved metrics
from local or mounted files without evaluation inputs. Publication validation
checks consistency; it does not prove which model generated the predictions.

Dataset and model rights remain separate from the source code's MIT license.
Consult the dataset card and [third-party notices](../THIRD_PARTY_NOTICES.md).
Downloaded data, weights, measurements, credentials, and generated reports stay
outside source commits; tracked definitions and synthetic fixtures remain code
repository assets.
