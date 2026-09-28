# S1MB project guidance

S1MB means **System One Mosaic Benchmark**. It combines specialized Choice,
Noul, and Score tasks. It does not establish unseen-task generalization or
training-data non-overlap. Communicate with the user in Japanese; write project
documentation, comments and docstrings in English.

## Structure and public source boundary

- `evaluator/`: Python package/CLI, adapters, scoring and benchmark definitions.
- `viewer/`: Next.js/TypeScript viewer and Storybook. `viewer/data` is a relative
  symlink to `../evaluator/data`; keep a single source of evaluation data.
- `evaluator/data/datasets/`, `evaluator/data/results/`, `evaluator/audits/`, and
  `tmp/` are local artifacts. Do not add datasets, weights, measurements, reports,
  or credentials to Git. Keep environment variable names in `.env.sample` only.
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
instructions. Check input lengths and reject overflow instead of silent truncation.

For local model inference on this workspace, use physical GPU 1 only
(`CUDA_VISIBLE_DEVICES=1`), inspect free memory, and run smoke checks first.
Do not fall back to CPU inference. Prefer supported FlashAttention 2 or SDPA.
Tokenization, data validation and unit tests may run on CPU.

Validate saved results before presenting them. Every JSON under a result root is
a result, so store auxiliary reports elsewhere. Keep incomplete runs visibly
incomplete. Use distinct run IDs for configurations; do not overwrite measurements.

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
