# S1MB — System One Mosaic Benchmark

S1MB evaluates typed decisions across **Choice**, **Noul**, and **Score** tasks.
It combines specialized benchmarks into a common evaluation and comparison
workflow. It does not establish general intelligence, unseen-task generalization,
or training-data non-overlap.

The repository contains a Python evaluator, model adapters, benchmark definitions,
shared scoring fixtures, and a Next.js results viewer. Measured benchmark results,
downloaded datasets, checkpoints, and development reports are not distributed here.

## Quick start

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), and Node.js 22.22.2
for the optional viewer. GPU-backed adapters require their own CUDA environment.

```sh
cd evaluator
uv sync --locked
uv run s1mb run --adapter dummy --category smoke-v1 --limit 2 --run-id smoke
```

The evaluator checks the latest revision of
[hotchpotch/s1mb-dataset](https://huggingface.co/datasets/hotchpotch/s1mb-dataset)
once at startup. It downloads and materializes data only when needed, then reads
all subsets locally. Every new result records the actual dataset commit SHA.
The dataset currently requires Hugging Face read access: authenticate using
`hf auth login` or set `HF_TOKEN`. Code access does not grant dataset access.
No credentials are required to run the public unit tests or build the viewer.

The dummy adapter produces **synthetic demo predictions**, not model measurements.
See [evaluator instructions](evaluator/README.md) for real adapters and full runs.
Never commit tokens; `.env.sample` lists the supported credential variables.

## View local results

After a dataset has been acquired through evaluation:

```sh
cd viewer
npm ci
npm run build
npm start
```

The viewer reads `viewer/data`, a symlink to `evaluator/data`, and loads local
results at startup. Restart it after new results or dataset updates. It binds to
the machine's Tailscale IPv4 address, or localhost if Tailscale is unavailable.
It does not expose raw inputs or prediction files as public assets.
See [viewer instructions](viewer/README.md) for additional result directories.

## Scores

| Task | Prediction | Primary metric | Better |
| --- | --- | --- | --- |
| Choice | Distribution over alternatives | Target mass at selected alternative | Higher |
| Noul | Probability of the authored true category | Brier score | Lower |
| Score | Distribution over numeric criteria | Normalized expected-score MAE | Lower |

Overview scores use a baseline-adjusted 0–100 scale where **higher is better**.
The overall index averages the three task scores equally. Zero means at or below
the task-specific baseline, not necessarily random predictions. Raw metrics remain
available. See [scoring definitions](evaluator/SCORING.md) for formulas and limits.

## Development

```sh
cd evaluator
uv sync --locked
uv run tox
```

```sh
cd viewer
npm ci
npm test
npm run typecheck
npm run build
```

Dataset integration tests run when the evaluation release is installed; otherwise
they are explicitly skipped. Set `S1MB_TEST_NO_DATASET=1` to exercise the public CI
configuration. Tests use synthetic fixtures; CI needs no private data, API tokens,
model checkpoints, or GPU. See [CONTRIBUTING.md](CONTRIBUTING.md).

See [release preparation](RELEASING.md) before distributing a source archive or
publishing a repository.

## License

Project code is available under the [MIT License](LICENSE). Dataset sources,
model weights, upstream inference packages, and bundled third-party material
retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
