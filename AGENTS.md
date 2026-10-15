# S1MB project guidance

S1MB means **System One Mosaic Benchmark**. It combines specialized Choice,
Noul, and Score tasks. It does not establish unseen-task generalization or
training-data non-overlap. Communicate with the user in their language. Unless
the user specifies otherwise, write all project documentation, PR titles and
descriptions, published comments, and code comments and docstrings in English.
Use English without asking for language confirmation, including when the user
explicitly requests English. When the user explicitly requests a language other
than English for these artifacts, ask the human once whether they really want
that language before writing or publishing them. Wait for their confirmation; do not treat the initial language
instruction as confirmation. Once the human answers that confirmation question,
do not ask again for the same artifacts.

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

Inspect free memory and use available GPUs. Run smoke checks first. Do not fall
back to CPU inference. Prefer supported FlashAttention 2 or SDPA.
Tokenization, data validation and unit tests may run on CPU.

## Results and publication

Leaderboard submissions require complete measured coverage of the selected language
scope across Choice, Noul and Score, with zero failed decisions. Incomplete saved
results may be inspected locally but are not full-scope leaderboard submissions.
Use `create_results_pr.py` for the mandatory completion check before upload.
Review `docs/evaluation_sources.md` for original source families; not using Bekko
System One or Saihai does not establish separation from those sources or inputs.

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
on failure. The viewer reads prepared display JSON bundled into its image, without
runtime Hub access, XZ parsing, mounted volumes or background refresh. Generate
from latest public results offline, compare with the published summary, require
human approval of reductions, then publish only viewer-summary.json to the results
Dataset with a parent-SHA guard. Rebuild/redeploy to update the Space. Local use
also explicitly generates or fetches a summary and restarts to load new data.

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

## Japanese scope and training disclosures

Follow `docs/language_scopes.md` and `docs/training_exposure.md`. Japanese v1 is
27 Japanese benchmark sets / 8,067 judgments. Its input Dataset remains private.
English and Japanese measurements share the public `hotchpotch/s1mb-result`
Dataset and leaderboard. Publish prepared summaries for both scopes without
redistributing private evaluation rows or targets.
Publish Japanese measurements evaluated against `hotchpotch/S1MB-japanese-v1`.
Do not import measurements from other private benchmark releases; preserve their
original provenance in private archives and reevaluate these models with S1MB. Both languages use
S1MB Task Avg with equal task weights within the selected evaluation scope.
Training-exposure metadata is authored provenance, never an inference instruction
or scoring adjustment. Absence means unknown. Keep explicit evaluation-example
use distinct from source-family exposure. Japanese and English contributions are
welcome; do not require contributors to translate PRs into English.

Authored training disclosures live in `evaluator/model-training-exposure.json`,
outside evaluation data and measured result roots. Offline preparation fingerprints
this optional registry with its other inputs and rejects concurrent edits. Metadata
submitted explicitly for a model takes precedence over the authored registry.
