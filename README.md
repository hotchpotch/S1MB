# S1MB — System One Mosaic Benchmark

### A benchmark for typed decisions across Choice, Noul, and Score tasks

**[S1MB Leaderboard](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard)** ·
[Explainer article](https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark/) ·
[Quick start](docs/quickstart.md) ·
[Results dataset](https://huggingface.co/datasets/hotchpotch/s1mb-result) ·
[Evaluation dataset](https://huggingface.co/datasets/hotchpotch/s1mb-dataset) ·
[Documentation](docs/README.md) ·
[Submit results](docs/contributing_results.md)

S1MB compares models that turn an input and a question into typed probabilities:
choosing among alternatives, judging an authored true/false condition, or scoring
against numeric criteria. It brings specialized datasets into one evaluation
workflow with shared input contracts, validation, and a leaderboard.

The **mosaic** is deliberate: different tasks test different abilities. The active
English category contains **137 benchmarks across 106 dataset subsets**. S1MB
supports model comparison within these tasks; it does not establish general
intelligence, unseen-task generalization, or training-data non-overlap.

## Highlights

- **Three decision types, one workflow.** Evaluate Choice, Noul, and Score while
  preserving structured inputs, authored criteria, and soft targets.
- **Comparable scores with visible detail.** Inspect raw task metrics alongside
  baseline-adjusted task scores and relative leaderboard rankings. Missing or
  failed measurements remain visibly incomplete.
- **Recorded evaluation conditions.** Results retain exact dataset and model
  revisions, evaluator provenance, and effective inference settings.
- **Local and hosted model interfaces.** Use supported adapters, including
  TypeSafe/Jev and Bekko, or implement the typed adapter contract for another model.
- **Community result submissions.** Validate and export measurements to a separate
  Hugging Face Dataset PR. Compare published entries and local runs in the viewer.

## Quick start

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). From a checkout of this repository:

```sh
cd evaluator
uv sync --locked
uv run s1mb run --adapter dummy --category smoke-v1 --limit 2 --run-id smoke-001
uv run s1mb validate data/results/smoke-001
```

This checks the evaluation pipeline using **synthetic predictions**, not a real
model. It downloads the configured evaluation dataset if needed; authenticate
with `uv run hf auth login` if dataset access requires it. Use a fresh run ID each
time. An online dataset check failure stops the run; `--offline-dataset` explicitly
uses an already installed dataset.

Continue with the [quick start](docs/quickstart.md) to open published results or
evaluate a real model. The [evaluation guide](docs/evaluation.md) covers runtime
setup, GPU selection, smoke checks, full runs, and validation.

## What is measured?

| Task | Model output | Primary metric | Better |
| --- | --- | --- | --- |
| Choice | Probabilities over alternatives | Target mass at the selected alternative | Higher |
| Noul | Probability of the authored true category | Brier score | Lower |
| Score | Probabilities over numeric criteria | Normalized expected-value MAE | Lower |

**Task Avg** combines baseline-adjusted task scores on a 0–100 scale. Adjustment
and clipping happen per benchmark, benchmarks are averaged within each task, and
the three tasks receive equal weight. Zero means at or below the reference
baseline; 100 is a reference ceiling. Noul adjustment uses balanced accuracy,
not Brier skill.

The viewer sorts by **Borda Score** by default. This is a relative ranking across
complete models, with equal weight per benchmark; it can change when the model
roster changes. It is distinct from Task Avg. Both require complete coverage in
the selected scope. See [benchmark scope](docs/benchmark_scope.md) and the
[scoring specification](evaluator/SCORING.md) for interpretation and limitations.

## Documentation

| I want to… | Start here |
| --- | --- |
| Install, try a smoke run, or open results | [Quick start](docs/quickstart.md) |
| Understand the tasks and evaluation boundaries | [Benchmark scope](docs/benchmark_scope.md) |
| Evaluate a model reproducibly | [Evaluation guide](docs/evaluation.md) |
| Add a model to the leaderboard | [Result submission](docs/contributing_results.md) |
| Compare local and published results | [Viewer guide](docs/viewer.md) |
| Integrate a model interface | [Adapter development](docs/adapters.md) |
| Find technical references and maintainer guides | [Documentation map](docs/README.md) |

To suggest a model, [open an evaluation request](https://github.com/hotchpotch/S1MB/issues/new?template=model_evaluation.yml).
Requests depend on volunteer time and resources; see the [request policy](docs/model_requests.md).
Adapter and code changes belong in source PRs; measurements belong in results
Dataset PRs.

## Development

The repository contains a Python evaluator and a Next.js/TypeScript viewer.
Use Python 3.11 for the evaluator and Node.js 22.22.2 for the viewer.

```sh
# From evaluator/
uv sync --locked
uv run tox
```

```sh
# From viewer/
npm ci
npm test
npm run typecheck
npm run build
```

Public checks use synthetic fixtures and require no private data, tokens, models,
or GPU. Start with [Contributing](CONTRIBUTING.md) and the
[developer workflow](docs/developer_workflow.md). Keep downloaded datasets,
checkpoints, credentials, generated results, and reports out of source commits.

## License

Project code is released under the [MIT License](LICENSE). Datasets, model
weights, upstream inference packages, and bundled third-party material retain
their own licenses and access terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
