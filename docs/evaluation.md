# Running evaluations for leaderboard submission

Use this guide to produce validated S1MB measurements. Then follow
[Adding a model to the leaderboard](contributing_results.md) to package results
and open a Hugging Face Dataset PR. Submission accepts both new benchmark files
and replacements; the original local evaluation runs remain separate.

S1MB evaluates specialized Choice, Noul, and Score tasks. It does not establish
unseen-task generalization or training-data non-overlap. Use the current dataset
schema, active benchmark definitions, and dataset-default instructions.

All commands in this guide run from `evaluator/`, except where a different working
directory is stated. Replace example paths and run IDs with your own values.

## 1. Prepare the evaluator and model

Use Python 3.11 and uv. From the project root:

```sh
cd evaluator
uv sync --locked
uv run hf auth login
uv run s1mb run --help
```

Dataset access is separate from code access. Authenticate with an account that
can read the repository configured in `dataset-source.json`. For API models,
configure credentials using the variable names in [`.env.sample`](../.env.sample).
Keep real credentials in a local ignored `.env`; do not include them in PRs.

Choose the adapter that implements the model's supported inference interface.
See [the evaluator guide](../evaluator/README.md#real-adapters),
[external adapter notes](../evaluator/OPEN_MODELS.md), and
[`upstream-models.json`](../evaluator/upstream-models.json). The base installation
does not include every model's runtime. Install the selected adapter's supported
dependencies and obtain any required checkpoint and upstream source checkout.
For example, Laya has the `laya` extra; several upstream adapters use the
`open-models` extra. Bekko requires its own compatible upstream environment.

Before evaluation, identify and record:

- The model/checkpoint and exact revision or API version.
- The adapter and upstream source revision, if applicable.
- Precision, attention backend, input limits, and batching settings.
- Known training/evaluation overlap and any model-specific input transformations.

For a new architecture, implement the interface in
[`adapters/base.py`](../evaluator/src/s1mb/adapters/base.py) and submit the adapter
as a code PR. An arbitrary text-generation model cannot use an existing typed
adapter merely by changing `--model`. Preserve criterion order, authored Noul
true/false definitions, structured inputs, soft targets, and numeric Score levels.
Keep targets and provenance out of model input. Validate probabilities and reject
input overflow instead of truncating.

## 2. Run a smoke check

First check the evaluator's plumbing if needed:

```sh
uv run s1mb run --adapter dummy \
  --category smoke-v1 --limit 2 --run-id plumbing-smoke-001
uv run s1mb validate data/results/plumbing-smoke-001
```

Dummy results are synthetic demos, not model measurements. A real-model smoke
check is still required. `smoke-v1` selects one benchmark for each task; `--limit 2`
restricts each benchmark to its first two cases. It is not a representative score
and does not establish that longer inputs in the complete set will fit.

### TypeSafe / Jev

With API credentials configured in the local `.env`:

```sh
uv run --env-file ../.env s1mb run --adapter typesafe --model jev \
  --category smoke-v1 --limit 2 --run-id jev-smoke-001
uv run s1mb validate data/results/jev-smoke-001
```

These are real API calls and may incur charges. Inspect the resolved model
version saved in the results. `jev` is an API selector; a leaderboard folder such
as `typesafe__jev_1_14` is a display identity, not necessarily an API model name.
Choose that folder only when the recorded version supports the name.

### Local GPU models, including Bekko

Inspect free memory before loading a model:

```sh
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
```

On this project's shared workspace, expose **physical GPU 1 only**, using the GPU
visibility prefix in the [Bekko example](../evaluator/README.md#bekko), before
running the commands below. That GPU becomes logical device `cuda:0`; `--device
cuda` uses it. On your own machine, choose the appropriate available GPU. Do not
fall back to CPU inference. Prefer FlashAttention 2 or SDPA where supported.

In the prepared Bekko environment, with S1MB importable and GPU visibility set:

```sh
python -m s1mb run \
  --adapter bekko --source /path/to/bekko-system-one --model /path/to/checkpoint \
  --device cuda --query-length 16384 --document-length 2048 \
  --microbatch-tokens 64000 \
  --category smoke-v1 --limit 2 --run-id bekko-smoke-001
python -m s1mb validate data/results/bekko-smoke-001
```

Inspect failures, probabilities, recorded model identity, and effective settings.
Lower Bekko's token batching budget if memory is insufficient; do not silently
shorten inputs. Flags are adapter-specific: consult `run --help` and the adapter
notes rather than applying a precision or attention option to every model.

## 3. Run the complete evaluation

After a successful smoke check, use `english-v1` without `--limit`. For example:

```sh
uv run --env-file ../.env s1mb run --adapter typesafe --model jev \
  --category english-v1 --run-id jev-full-001
```

For Bekko, retain the tested runtime and input settings:

```sh
python -m s1mb run \
  --adapter bekko --source /path/to/bekko-system-one --model /path/to/checkpoint \
  --device cuda --query-length 16384 --document-length 2048 \
  --microbatch-tokens 64000 \
  --category english-v1 --run-id bekko-full-001
```

Every run needs a fresh ID. `s1mb run` does not resume into an existing run
directory or overwrite it. Save exact commands and environment notes outside
result roots, for example in `../tmp/my-model/evaluation-notes.md`.

To inspect the required set after acquiring the dataset:

```sh
uv run s1mb list --category english-v1
uv run s1mb check-data --category english-v1
```

Use `--task choice`, `--task noul`, or `--task score` for intentional task subsets.
Use repeated `--benchmark BENCHMARK_ID` options to rerun selected benchmarks from
the category. Give each rerun a new ID, then export the intended replacement files
into the same published model folder. Do not select different configurations per
benchmark without documenting that choice in the submission.

Input overflow and model failures must remain visible. Address the underlying
issue and rerun, or submit the available coverage as incomplete. A run can leave
saved files even if its process exits unsuccessfully; inspect and validate them.

## Dataset revisions

Before loading a model, `s1mb run` resolves the configured Hub revision once,
downloads only when needed, and locks the installed dataset for the run. Each
result records its exact dataset repo ID and commit SHA. An online check failure
stops the run; it never silently switches to cached data.

`--revision` on `s1mb run` is a **model** revision option, not a dataset revision
selector. Select a dataset revision in `evaluator/dataset-source.json`. Use a full
commit SHA when reproducing a particular evaluation release. `--offline-dataset`
explicitly skips the online check and uses the installed dataset and its recorded
provenance; it does not fetch missing data.

The importer rejects changes to existing rows instead of silently replacing them.
For a different historical dataset release, use an isolated checkout/data root
with matching benchmark/category definitions. An alternate `--data-dir` root
must contain the `benchmarks/` and `categories/` definitions before acquisition;
place that global option before the subcommand. Keep the original data available
until its raw results have been validated and exported.

Published model folders may mix dataset revisions. Synchronization materializes
recorded releases for validation and baseline calculation; it does not rewrite
all results as if they came from the latest release.

## 4. Validate, inspect, and submit

For a raw run against its installed evaluation data:

```sh
uv run s1mb validate data/results/jev-full-001
```

Validation recomputes metrics from predictions and checks input hashes, IDs,
probabilities, counts, completion status, and consistency within the run. A valid
partial result is still partial. Check completed coverage against `s1mb list`;
validation success alone is not evidence of complete category coverage.

Optional local preview, from `viewer/`:

```sh
npm ci
npm run build
npm start -- --results-dir ../evaluator/data/results/jev-full-001
```

The viewer requires Node.js 22.22.2; compressed published results also require
`xz` (`xz-utils` on Debian/Ubuntu). The wrapper binds to the machine's Tailscale
IPv4 address or localhost. Restart after changing data or results. Selecting the
raw run explicitly avoids accidentally inspecting a previously synchronized
Hub snapshot.

Overview task scores and the overall summary use baseline-adjusted scores on a
0–100 scale, with higher values better and complete coverage required. Raw Choice
target mass is higher-better; raw Noul Brier and Score normalized expected-value
MAE are lower-better. `1 - MAE` is not accuracy. Consult
[the scoring specification](../evaluator/SCORING.md) for formulas and exclusions.

Continue with [the submission guide](contributing_results.md#prepare-a-submission)
for `metadata.json`, compressed export, the PR description template, and upload.
Keep measurements, downloaded data, checkpoints, and credentials out of the code
repository. Dataset and model licenses remain separate from the code's MIT license.
