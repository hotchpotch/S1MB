# Documentation map

Start with the [quick start](quickstart.md) for installation, a pipeline smoke
check, or a local leaderboard. Read [benchmark scope](benchmark_scope.md) to
understand the three decision types and what the scores can establish.

## Workflow routes

| Goal | Route |
| --- | --- |
| Inspect published measurements | [Quick start](quickstart.md#open-published-results) → [Viewer guide](viewer.md) |
| Evaluate and submit a model | [Evaluation](evaluation.md) → [Result submission](contributing_results.md) |
| Add a model interface | [Adapter development](adapters.md) → [Runtime reference](../evaluator/README.md#real-adapters) → [Evaluation](evaluation.md) |
| Interpret a leaderboard | [Benchmark scope](benchmark_scope.md) → [Scoring specification](../evaluator/SCORING.md) |
| Change evaluator or viewer code | [Contributing](../CONTRIBUTING.md) → [Developer workflow](developer_workflow.md) |
| Suggest a model for volunteers | [Model evaluation requests](model_requests.md) |
| Deploy the maintained viewer | [Space deployment](huggingface_space_deploy.md) |
| Publish a source release | [Release checklist](../RELEASING.md) |

## Authoritative references

Keep operational details in the guide that owns them; link to it from introductions
and component READMEs instead of maintaining duplicate procedures.

| Topic | Source of truth |
| --- | --- |
| Task meaning, category scope, and interpretation limits | [Benchmark scope](benchmark_scope.md) |
| Setup, smoke/full runs, reruns, and dataset revisions | [Evaluation](evaluation.md) |
| Result layout, metadata, export, validation, Dataset PRs, and synchronization | [Result submission](contributing_results.md) |
| Local viewer startup, source selection, Docker, and troubleshooting | [Viewer guide](viewer.md) |
| Inference interface and integration requirements | [Adapter development](adapters.md) |
| Supported runtimes and model-specific settings | [Evaluator reference](../evaluator/README.md) and [external model notes](../evaluator/OPEN_MODELS.md) |
| Metrics, baseline eligibility, Task Avg, and Borda | [Scoring specification](../evaluator/SCORING.md) |
| Display generation, reduction review, and image bundling | [Prepared display data](../viewer/DISPLAY_DATA.md) |
| Source checks and generated-file conventions | [Developer workflow](developer_workflow.md) |
| Deployment permissions, branches/worktrees, workflow, and volumes | [Space deployment](huggingface_space_deploy.md) |
| Environment configuration names | [`.env.sample`](../.env.sample) |

The [evaluator](../evaluator/README.md) and [viewer](../viewer/README.md) READMEs
are component entry points. [Historical open-model evaluation notes](../evaluator/OPEN_MODEL_RESULTS.md)
describe earlier runs and their limitations; they are not the current leaderboard
or current coverage specification.

## Source, data, and publication boundaries

| Material | Location | Publication path |
| --- | --- | --- |
| Code, benchmark/category definitions, synthetic test fixtures | This repository | Source PR |
| Evaluation inputs and targets | [Evaluation dataset](https://huggingface.co/datasets/hotchpotch/s1mb-dataset) | Dataset release; consult its card for access and terms |
| Model measurements and display metadata | [Results dataset](https://huggingface.co/datasets/hotchpotch/s1mb-result) | Validated results Dataset PR |
| Downloaded data, weights, raw runs, caches, reports, credentials | Ignored local storage | Keep out of source commits |

`viewer/data` is a relative symlink to `../evaluator/data`, preserving one local
source for definitions and results. The viewer reads saved metrics without
evaluation inputs; prediction validation remains the evaluator's responsibility.
The source code's MIT license does not grant dataset or checkpoint redistribution
rights. See [third-party notices](../THIRD_PARTY_NOTICES.md).
