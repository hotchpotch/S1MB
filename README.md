# S1MB — System One Mosaic Benchmark

### Comparing System One Decision Models across 100+ benchmarks

**[S1MB Leaderboard](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard)**
([en-v1](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard?category=english-v1) ·
[ja-v1](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard?category=japanese-v1) ·
[vision-v1](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard?category=vision-v1)) ·
[Explainer article](https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark/) ·
[Quick start](docs/quickstart.md) ·
[Results dataset](https://huggingface.co/datasets/hotchpotch/s1mb-result) ·
[English evaluation dataset](https://huggingface.co/datasets/hotchpotch/S1MB-english-v1) ·
[Documentation](docs/README.md) ·
[Submit results](docs/contributing_results.md)

S1MB provides a practical reference point for comparing **System One Decision
Models** on the same inputs and metrics. These models use text, images and instructions
to select an option (**Choice**), make a yes/no judgment (**Noul**), or assign a
numeric score using supplied criteria (**Score**). The leaderboard lets you
compare overall results and inspect which tasks each model handles well or
struggles with.

S1MB supports **English text (`english-v1`), Japanese text (`japanese-v1`), and
vision (`vision-v1`)** evaluation. Select a scope with `s1mb run --category`.
Use `s1mb run --all` for the English/Japanese text union; select vision explicitly
with `--category vision-v1`.

**Vision v1** contains **26 image benchmarks / 7,800 judgments**, covering
image-bearing Choice, Noul, Score and optional bounding-box localization (Box).
Ranking requires all 20 non-Box benchmarks / 6,000 judgments; the six optional
Box benchmarks are reported separately. Its
[evaluation Dataset](https://huggingface.co/datasets/hotchpotch/S1MB-vision-v1)
is private and requires authorized access. Supported image adapters include
TypeSafe/Jev, OpenAI Decisions and Imajev; models without Box support use
`--exclude-box`. Follow the [vision evaluation workflow](docs/vision-evaluation.md)
for installation, adapter capabilities, smoke/full runs and submission.

The **mosaic** combines tasks from existing public NLP datasets with synthetic
tasks in a shared evaluation workflow. English v1 contains **137 benchmarks
across 106 dataset subsets**, including six synthetic benchmarks that probe how
models respond to varied instructions, contexts, and decision criteria.
Japanese v1 contains **27 Japanese benchmarks / 8,067 judgments**.
The leaderboard selects the evaluation scope through its language controls or
`?category=english-v1` / `?category=japanese-v1` query parameters. Both languages
use the public [results Dataset](https://huggingface.co/datasets/hotchpotch/s1mb-result)
and the same leaderboard. The
[Japanese evaluation Dataset](https://huggingface.co/datasets/hotchpotch/S1MB-japanese-v1)
remains private and requires authorized access to run evaluations or validate
predictions against its targets. See [language scopes](docs/language_scopes.md).

S1MB offers a starting point for model comparison and improvement within these
tasks. Its scores do not establish unseen-task generalization or training-data
non-overlap. See the [explainer article](https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark/)
for the motivation, example results, and limitations, or [submit your own model's
results](docs/contributing_results.md) to extend the comparison.

## Highlights

- **Three decision types, one workflow.** Evaluate Choice, Noul, and Score while
  preserving structured inputs, authored criteria, and soft targets.
- **Text and vision evaluation.** Compare image-bearing decisions in a separate
  Vision v1 scope, with optional Box localization reported independently.
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
For image-bearing models, start with the [vision workflow](docs/vision-evaluation.md)
and [vision adapter examples](docs/vision-adapters.md).

## What is measured?

| Task | Model output | Primary metric | Better |
| --- | --- | --- | --- |
| Choice | Probabilities over alternatives | Target mass at the selected alternative | Higher |
| Noul | Probability of the authored true category | Brier score | Lower |
| Score | Probabilities over numeric criteria | Normalized expected-value MAE | Lower |
| Box (vision, optional) | Normalized full-image bounding box | Mean IoU | Higher |

**Task Avg** combines baseline-adjusted task scores on a 0–100 scale. Adjustment
and clipping happen per benchmark, benchmarks are averaged within each task, and
the three tasks receive equal weight. Zero means at or below the reference
baseline; 100 is a reference ceiling. Noul adjustment uses balanced accuracy,
not Brier skill.
Optional Box results do not contribute to Task Avg or Borda Score.

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
| Evaluate a vision model | [Vision evaluation workflow](docs/vision-evaluation.md) |
| Add a model to the leaderboard | [Result submission](docs/contributing_results.md) |
| Evaluate models and submit results with an agent (explicit invocation) | [$s1mb-evaluate-and-submit](.agents/skills/s1mb-evaluate-and-submit/SKILL.md) |
| Compare local and published results | [Viewer guide](docs/viewer.md) |
| Integrate a model interface | [Adapter development](docs/adapters.md) |
| Implement image input or Box support | [Vision adapter examples](docs/vision-adapters.md) |
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

## Acknowledgments

As its name, **System One Mosaic Benchmark**, suggests, S1MB brings together
many existing benchmarks. We thank the researchers, annotators, dataset creators
and maintainers whose work makes this mosaic possible. The individual evaluation
Dataset cards link to their upstream datasets, source projects and attribution
notices: [English v1](https://huggingface.co/datasets/hotchpotch/S1MB-english-v1#sources),
[Japanese v1](https://huggingface.co/datasets/hotchpotch/S1MB-japanese-v1#sources-and-licenses)
and [Vision v1](https://huggingface.co/datasets/hotchpotch/S1MB-vision-v1#sources-and-licenses).
The [source checklist](docs/evaluation_sources.md) provides another route to those
works. Please credit and cite the original research alongside S1MB when using
its derived evaluations.

### Vision

We thank the teams behind [MTEB](https://github.com/embeddings-benchmark/mteb),
[MIEB](https://arxiv.org/abs/2504.10471), and
[MMEB / VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec)
([MMEB evaluation dataset](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval)).
Their work helped us discover candidate tasks and consider broad coverage of
visual capabilities. Vision v1 adapts original image benchmarks to typed
decisions and adds authored synthetic tasks; its protocols and scores differ
from embedding evaluation. We also thank the original benchmark authors and
image contributors, linked in the [vision source checklist](docs/evaluation_sources.md#vision-sources)
and [source attribution and license inventory](docs/vision-source-licenses.md).

## Citation

If you use S1MB in your research, please cite:

```bibtex
@misc{tateno2026s1mb,
  author       = {Tateno, Yuichi},
  title        = {{S1MB}: System One Mosaic Benchmark},
  year         = {2026},
  howpublished = {\url{https://github.com/hotchpotch/S1MB}}
}
```

For reproducibility, also report the evaluation scope, dataset repository
and exact revision, and evaluator version or commit used in your experiments.
When using benchmarks derived from existing datasets, also cite the relevant
original sources listed in [Evaluation sources](docs/evaluation_sources.md).

## License

Project code is released under the [MIT License](LICENSE). Datasets, model
weights, upstream inference packages, and bundled third-party material retain
their own licenses and access terms. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Language scopes and training disclosure

See [language scopes](docs/language_scopes.md) for English/Japanese datasets, language
switching, `s1mb run --all`, migration provenance and shared result publication.
See [training exposure](docs/training_exposure.md) for model metadata and known-domain
badges. **Known domain** denotes disclosed source-family or related-domain training;
it does not establish use of S1MB evaluation examples or an independent record-level
overlap audit. Exact training revisions and splits may be undisclosed; consult the
linked evidence and limitations. Actual evaluation-example use is labeled separately.
Japanese and English contributions and PR descriptions are welcome.

For training models to improve Japanese typed decisions, use the
[Saihai training Dataset](https://huggingface.co/datasets/hotchpotch/saihai-v1-dataset)
with authorized access and follow its source-specific licenses. It includes
training examples from ten source families represented in `S1MB-japanese-v1`.
The maintainer reports removing duplicate examples and evaluation overlaps during
preparation; this does not make those source domains unseen. When submitting a
model trained on these examples to S1MB, declare the applicable **Known domains**
in `metadata.json` using `training_exposure` and `same_source_family`. Record the
exact training revision and effective configurations, including any filtering.
See [training disclosure requirements](docs/training_exposure.md#training-for-japanese-decisions).
