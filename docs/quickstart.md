# Quick start

Choose a path: inspect published measurements, check the evaluation pipeline, or
run a supported model. For task definitions and comparison limits, read
[benchmark scope](benchmark_scope.md).

## Prerequisites

- Evaluator and result synchronization: Python 3.11 and [uv](https://docs.astral.sh/uv/).
- Viewer: Node.js 22.22.2, npm, and `xz` (`xz-utils` on Debian/Ubuntu).
- Evaluation or publication validation: access to the configured evaluation
  dataset. Source code access does not grant dataset access.

Commands below assume a checkout of this repository. Run Python commands from
`evaluator/` and npm commands from `viewer/`. Credential settings are listed in
[`.env.sample`](../.env.sample); keep real values in an ignored local `.env`.

## Open published results

From the repository root:

```sh
cd evaluator
uv sync --locked
uv run s1mb sync-results --repo-id hotchpotch/s1mb-result
cd ../viewer
npm ci
npm run prepare-display -- --results-dir ../evaluator/data/hub-results
npm run build
npm start
```

Open the URL printed by the start wrapper. It binds to the machine's Tailscale
IPv4 address when available, otherwise localhost. The public results repository
contains measurements; synchronization validates them against their recorded
evaluation dataset revisions and may require dataset authorization. If required,
run `uv run hf auth login` from `evaluator/`, then retry synchronization.

The viewer itself requires no evaluation inputs or Python runtime. If you already
have a fully downloaded, validated results folder, skip synchronization and pass
`npm run prepare-display -- --results-dir /absolute/path/to/results`, then
`npm start` after installation and build.
Git/Xet pointer files are not usable results. See the [viewer guide](viewer.md)
for multiple sources, updates, Docker, and troubleshooting.

## Check the evaluation pipeline

From the repository root:

```sh
cd evaluator
uv sync --locked
uv run s1mb run --adapter dummy --category smoke-v1 --limit 2 --run-id smoke-001
uv run s1mb validate data/results/smoke-001
```

The dummy adapter returns synthetic probabilities. This checks dataset loading,
inference plumbing, result storage, and validation; it does not measure a model.
`smoke-v1` selects one benchmark per task, limited here to two cases each. These
results are deliberately partial and do not qualify for leaderboard ranking.

Every run requires a new ID. Output goes to ignored `evaluator/data/results/<run-id>/`.
The run checks the configured Hub revision once and records its exact SHA. A failed
online check stops evaluation. `--offline-dataset` explicitly uses installed data;
it cannot acquire missing data. See [dataset revisions](evaluation.md#dataset-revisions).

## Evaluate a real model

Select an adapter from the [runtime reference](../evaluator/README.md#real-adapters).
For example, with TypeSafe API credentials configured in the local `.env`, run
from `evaluator/`:

```sh
uv run --env-file ../.env s1mb run --adapter typesafe --model jev \
  --category smoke-v1 --limit 2 --run-id jev-smoke-001
uv run s1mb validate data/results/jev-smoke-001
```

These are real API calls and may incur charges. Inspect failures, saved model
identity, and effective settings before continuing. After a successful smoke
check, evaluate the full category with a fresh ID:

```sh
uv run --env-file ../.env s1mb run --adapter typesafe --model jev \
  --category english-v1 --run-id jev-full-001
uv run s1mb validate data/results/jev-full-001
```

Local model adapters require their supported runtime and an explicitly selected
GPU. Follow the [evaluation guide](evaluation.md) before loading a model; a small
smoke run does not establish that full-length inputs will fit.

## Preview and submit

After installing and building the viewer, run from `viewer/`:

```sh
npm run prepare-display -- --results-dir ../evaluator/data/results/jev-full-001
npm start
```

Validation success does not imply complete coverage. Inspect missing or failed
benchmarks, then follow [result submission](contributing_results.md) to create
metadata, run `export-results` and `validate-results`, and open a Dataset PR.
Keep original local runs intact and submit source changes separately.
