# S1MB project guidance

S1MB means **System One Mosaic Benchmark**. It combines specialized Choice,
Noul, and Score tasks. It does not establish unseen-task generalization or
training-data non-overlap. Communicate in the user's language; write project
documentation, comments and docstrings in English.

## Structure and public source boundary

- `docs/evaluation.md`: evaluation setup, smoke checks, full runs and validation.
- `docs/contributing_results.md`: authoritative result submission and Dataset PR guide.
- `evaluator/`: Python package/CLI, adapters, scoring and benchmark definitions.
- `viewer/`: Next.js/TypeScript viewer and Storybook. `viewer/data` is a relative
  symlink to `../evaluator/data`; keep a single source of evaluation data.
- Local artifacts include `evaluator/data/datasets/`, `evaluator/data/results/`,
  `evaluator/data/hub-results`, `evaluator/data/.hub-results*`,
  `evaluator/data/result-datasets/`, `evaluator/audits/`, `tmp/`, `output/`, and
  `evaluator/output/`. Keep these ignored; do not add datasets, weights,
  measurements, reports, or credentials to the source repository. Keep environment
  variable names in `.env.sample` only.
- Keep benchmark/category definitions and synthetic test fixtures tracked. Do not
  ignore all of `evaluator/data/` or blanket-ignore JSON/XZ files.
- Result files belong in the separate HF results dataset; adapter/code changes
  belong in source PRs. Follow `RELEASING.md` for release contents and history.
- Project code is MIT. Preserve separate third-party licenses; do not imply that
  the code license grants dataset or checkpoint redistribution rights.

Keep implementation focused on the current schema and interfaces. Remove obsolete
compatibility paths and experiment-only branches rather than maintaining aliases.
Atomic temporary files, bounded batching, input validation and operational error
handling are production behavior and must remain reliable.

## Data and inference

`s1mb run` checks the latest configured Hub revision once, downloads only if needed,
and locks that installed dataset for the run. Record its exact SHA in results.
`--offline-dataset` explicitly skips the online check; never silently fall back to
stale data. Use `dataset-source.json` for source configuration.

Only the active manifest controls membership. Keep benchmark definitions and
categories consistent with it. Read current `input` and `targets` by stable ID;
preserve declared criterion order, authored Noul definitions, structured values,
soft targets and numeric Score levels. Ranking distributions are not Score labels.
Keep targets, provenance and identifiers out of model text. Use dataset-default
instructions. Check input lengths and reject overflow instead of silent truncation,
except Bekko v0 evaluations explicitly use native adaptive budgeting and record
its truncation policy in model metadata.

On the maintainers' shared workspace, use physical GPU 1 only
(`CUDA_VISIBLE_DEVICES=1`), inspect free memory, and run smoke checks first.
On other machines, select an available GPU explicitly. Do not fall back to CPU
inference. Prefer supported FlashAttention 2 or SDPA.
Tokenization, data validation and unit tests may run on CPU.

## Results and publication

Validate saved results before presenting them. Local `s1mb run` output goes to
`evaluator/data/results/<run-id>/`; use a fresh run ID and preserve original local
measurements. Store reports and auxiliary JSON outside result roots.

The HF results layout is `<org-or-user>__<model-id>/metadata.json` plus
`<benchmark-id>.json.xz`. A model folder is one leaderboard row. `metadata.json`
is the only display-metadata exception to the result-file rule. Published
benchmark files may be added or overwritten through a Dataset PR; HF history
preserves prior versions. Separate model versions/configurations can use new IDs.

Allow different original run IDs, evaluator versions and dataset revisions in a
published model folder. Preserve each result's dataset repo ID and exact SHA,
model revision, and evaluator provenance. Validate and compute baselines against
that result's recorded dataset revision, not automatically against the latest.
Current benchmark/category definitions control leaderboard membership. Do not
reject a result solely because its dataset revision differs.

Use `export-results` to package results, `validate-results` for published folders,
and `sync-results` to install a verified Hub snapshot. Keep incomplete results
visibly incomplete. Synchronization must leave the installed snapshot unchanged
on failure. The viewer reads saved metrics from local or mounted result files without evaluation
inputs. Publication validation remains the evaluator's responsibility. On requests,
a separate Node process checks filesystem metadata (every access locally, at most
hourly in Spaces), reads changed files, and swaps summaries atomically. Existing
requests receive cached data immediately; refresh failures preserve that cache.
An external mount or synchronization process must make Hub updates visible locally.

## Score presentation

Overview task scores and overall summaries use baseline-adjusted scores multiplied
by 100: higher is better. Adjust and clip per benchmark, average benchmarks within
each task, then weight the three task scores equally. Zero means at or below the
reference baseline; 100 is a reference ceiling. Require complete coverage.
Noul's adjusted score uses balanced accuracy, not Brier skill.

Details retain primary metrics and directions: Choice target mass is higher-better;
Noul Brier and Score normalized expected-value MAE are lower-better. Complements
such as `1 - MAE` are not accuracy. Follow `evaluator/SCORING.md` and keep Python
and viewer formulas aligned through shared synthetic fixtures.

## Checks and services

Run Python commands from `evaluator/`, npm commands from `viewer/`, and follow local
guidance. Public tests/CI must run without private data, tokens, models or GPU.
Mark dataset integration tests explicitly. Restart the viewer after data/results
change. Bind HTTP only to the machine's Tailscale IPv4 address or localhost.

The HF Space image's dedicated bootstrap may bind to `0.0.0.0:7860` inside the
managed Space container. This exception does not apply to local services or the
normal viewer start wrapper. Follow `docs/huggingface_space_deploy.md` for deployment.
