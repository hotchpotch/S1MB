# S1MB — System One Mosaic Benchmark

S1MB evaluates typed decisions across **Choice**, **Noul**, and **Score** tasks.
It combines specialized benchmarks into a common evaluation and comparison
workflow. It does not establish general intelligence, unseen-task generalization,
or training-data non-overlap.

This repository contains the Python evaluator, model adapters, benchmark
definitions, shared scoring fixtures, and a Next.js leaderboard viewer. Model
measurements are submitted separately through Hugging Face Dataset pull requests.
Downloaded datasets, checkpoints, credentials, and generated results are not
included in the source distribution.

## Guides

- [Developer workflow](docs/developer_workflow.md): worktrees, branch synchronization,
  local results, checks, and authorized HF deployment.

- [Run evaluations](docs/evaluation.md): setup, smoke checks, full runs, and validation.
- [Add leaderboard results](docs/contributing_results.md): model metadata, compressed
  results, Dataset PRs, replacements, and synchronization.
- [Scoring](evaluator/SCORING.md): primary metrics and baseline-adjusted summaries.
- [Evaluator reference](evaluator/README.md) and [adapter notes](evaluator/OPEN_MODELS.md).
- [Viewer reference](viewer/README.md), [code contributions](CONTRIBUTING.md), and
  [source release preparation](RELEASING.md).

## Quick start

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
cd evaluator
uv sync --locked
uv run s1mb run --adapter dummy --category smoke-v1 --limit 2 --run-id smoke-001
uv run s1mb validate data/results/smoke-001
```

The dummy adapter produces **synthetic demo predictions**, not model measurements.
It checks evaluation plumbing using the evaluation dataset. For a real model,
follow the [evaluation guide](docs/evaluation.md); local GPU adapters need their
supported model runtime, and API models need provider credentials.

`s1mb run` resolves the revision configured in
[`evaluator/dataset-source.json`](evaluator/dataset-source.json) once, downloads
only when needed, and records the exact dataset commit SHA in every result.
If the configured dataset requires authorization, run `uv run hf auth login`
from `evaluator/` before evaluation. Code access does not grant dataset access.
Credential configuration is documented in [`.env.sample`](.env.sample); keep real
credentials in an ignored local `.env`.

A failed online check stops evaluation. `--offline-dataset` explicitly uses an
already installed dataset. Use a fresh `--run-id` for each local run. The standard
output is `evaluator/data/results/<run-id>/`, which is ignored by Git.

## View results

Use Node.js 22.22.2 and npm. Once local or mounted result files are
available, run these commands from the repository root:

```sh
cd viewer
npm ci
npm run build
npm start
```

The viewer reads `viewer/data`, a relative symlink to `evaluator/data`. It selects
synchronized `data/hub-results` when present, otherwise `data/results`. To inspect
a particular local run, pass `--results-dir ../evaluator/data/results/RUN_ID`.
Results load on first access; subsequent requests trigger background filesystem
checks and keep serving the current cache while changed files are loaded.

To use contributed results, run `uv run s1mb sync-results --repo-id hotchpotch/s1mb-result`
from `evaluator/`. The results repository is public. If synchronization creates
`data/hub-results` for the first time, restart the viewer to select it. Later
updates to the selected source are picked up by background checks.
Reading `.json.xz` files requires `xz`
(`xz-utils` on Debian/Ubuntu). See the [submission guide](docs/contributing_results.md)
for PR previews and recorded-dataset validation.

The start wrapper binds only to the machine's Tailscale IPv4 address, or localhost
when Tailscale is unavailable. The browser receives summaries; raw inputs and
prediction files are not served as public assets.

## Scores

| Task | Prediction | Primary metric | Better |
| --- | --- | --- | --- |
| Choice | Distribution over alternatives | Target mass at selected alternative | Higher |
| Noul | Probability of the authored true category | Brier score | Lower |
| Score | Distribution over numeric criteria | Normalized expected-value MAE | Lower |

Overview task scores use a baseline-adjusted 0–100 scale, where **higher is better**.
Scores are adjusted and clipped per benchmark, then averaged within each task.
The overall index weights the three tasks equally and requires complete coverage.
Zero means at or below the reference baseline; 100 is a reference ceiling.
Raw metrics retain their original directions. See [scoring](evaluator/SCORING.md)
for formulas, exclusions, and interpretation limits.

## Contribute leaderboard results

Each model has a folder containing `metadata.json` and per-benchmark `.json.xz`
files. Add benchmarks or replace existing files through a Dataset PR. Different
recorded dataset revisions can coexist; each result is validated and scored
against its own revision. Use a new model ID for versions or configurations that
should appear as separate rows.

The [submission guide](docs/contributing_results.md) covers `export-results`,
`validate-results`, PR creation and updates, and `sync-results`. Merging a Dataset
PR becomes visible after synchronization (or the managed Dataset mount updates)
and a request triggers the next background filesystem check.
Adapter or other source changes belong in a separate code PR.

## Development and generated files

Run Python commands from `evaluator/`:

```sh
uv sync --locked
uv run tox
```

Run npm commands from `viewer/`:

```sh
npm ci
npm test
npm run typecheck
npm run build
```

Public tests and builds need no private data, tokens, models, or GPU. Dataset
integration tests are marked explicitly and skip when data is absent. The
[CI workflow](.github/workflows/check.yml) also runs without downloaded datasets.

Standard evaluation results, synchronized Hub results and snapshots, cached
historical datasets, audits, and `tmp/` staging files are ignored by Git.
`output/` at the repository root and `evaluator/output/` are also ignored for
explicit `--output` use. Other custom destinations need their own ignore rule
or must be outside the repository. Benchmark definitions and synthetic test
fixtures remain source files; do not ignore or remove them as generated results.

Before publishing source, follow [RELEASING.md](RELEASING.md), including review of
the actual release file list and Git history. Ignore rules do not remove files
that were already committed.

## License

Project code is available under the [MIT License](LICENSE). Dataset sources,
model weights, upstream inference packages, and bundled third-party material
retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
