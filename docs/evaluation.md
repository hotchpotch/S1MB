# Running evaluations for leaderboard submission

Use this guide to produce validated S1MB measurements. For a first pipeline check
or a local leaderboard, start with the [quick start](quickstart.md). Then follow
[Adding a model to the leaderboard](contributing_results.md) to package results
and open a Hugging Face Dataset PR. Submission accepts both new benchmark files
and replacements; the original local evaluation runs remain separate.

S1MB evaluates specialized Choice, Noul, and Score tasks. It does not establish
unseen-task generalization or training-data non-overlap. Use the current dataset
schema, active benchmark definitions, and dataset-default instructions.

All commands in this guide run from `evaluator/`, except where a different working
directory is stated. Replace example paths and run IDs with your own values.

The configured [evaluation dataset](https://huggingface.co/datasets/hotchpotch/s1mb-dataset)
contains the benchmark inputs and targets; published measurements belong in the
separate [results dataset](https://huggingface.co/datasets/hotchpotch/s1mb-result).
Consult the dataset card for access, provenance and license terms. For a new
model interface, follow [adapter development](adapters.md); for display setup,
see [the viewer guide](viewer.md).

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
Additional English typed-decision runtimes and their input limits are documented
in [external model conditions](../evaluator/OPEN_MODELS.md#additional-english-model-runtimes).
Winnow also requires a separately built CUDA server; its `--server-host` must be
the machine's Tailscale IPv4 address or localhost when unavailable, and
`--server-port` must be unused. APUS needs its own pinned Transformers environment.
For example, Laya has the `laya` extra; several upstream adapters use the
`open-models` extra. Kev and Open-Jev also accept `--case-batch-size` (default 16);
use 1 for their audited 9B conditions to avoid the cross-case BF16 drift observed
in smoke comparisons. CLM and Tev require `--case-batch-size 1`; see
[their input and probability conditions](../evaluator/OPEN_MODELS.md#clm-and-tev)
before comparing measurements. Both use `--source` and a pinned `--revision`.
Bekko uses the `bekko-v0` extra and adapter with a Hugging Face model ID;
its remote `BekkoSentenceTransformer.predict()` supplies inference.
Meta Encoder uses the `meta-encoder` extra and requires an explicit positive
cosine-softmax `--temperature`. Its text-choice input limit, candidate caching,
overflow rejection, and input rendering are documented in
[the adapter notes](../evaluator/OPEN_MODELS.md#meta-encoder).

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
input overflow instead of truncating unless an explicitly documented adapter
policy applies. Bekko v0 uses native adaptive budgeting with truncation recorded
in model metadata; use a Hub model containing the current runtime.

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

### Bekko standalone CPU smoke

The `bekko-v0` adapter supports an explicit CPU diagnostic when a GPU is
unavailable. Follow the [standalone v0 instructions](../evaluator/README.md#bekko)
to install dependencies and evaluate the remote Hub model. CPU runs require
`--device cpu --category smoke-v1 --limit 1` (or `2`). Validate the saved partial
results. Full evaluation still requires the GPU path; CPU smoke timing does not
predict GPU throughput.

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

With the `bekko-v0` extra installed and GPU visibility set:

```sh
uv run --extra bekko-v0 s1mb run \
  --adapter bekko-v0 --model hotchpotch/bekko-system-one-v0-17m \
  --device cuda --context-limit 7999 --document-length 3800 \
  --microbatch-tokens 64000 \
  --category smoke-v1 --limit 2 --run-id bekko-smoke-001
uv run s1mb validate data/results/bekko-smoke-001
```

Inspect failures, probabilities, recorded model identity, and effective settings.
Lower Bekko's token batching budget if memory is insufficient. Standalone v0
uses its documented adaptive input budget independently of this batching budget. Flags are adapter-specific: consult `run --help` and the adapter
notes rather than applying a precision or attention option to every model.

## 3. Run the complete evaluation

After a successful smoke check, use `english-v1` without `--limit`. For example:

```sh
uv run --env-file ../.env s1mb run --adapter typesafe --model jev \
  --category english-v1 --run-id jev-full-001
```

For Bekko, retain the tested runtime and input settings:

```sh
uv run --extra bekko-v0 s1mb run \
  --adapter bekko-v0 --model hotchpotch/bekko-system-one-v0-17m \
  --device cuda --context-limit 7999 --document-length 3800 \
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
npm run prepare-display -- --results-dir ../evaluator/data/results/jev-full-001
npm start
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

### Generalization-only runs

Add `--generalization-only` to `s1mb run` to evaluate only the active category's
generalization benchmarks (currently six in `english-v1`). This intersects with
`--task` and repeated `--benchmark` filters; an empty selection is an error.
Use a fresh run ID. These runs retain normal per-benchmark completeness, but do
not provide full-category coverage. The viewer lists them when **Generalization tasks only** is checked and all six
benchmarks are complete. The default leaderboard requires full-category coverage.


Capacity-limited decision adapters can be retried under explicitly extended
conditions. See [capacity extensions](../evaluator/OPEN_MODELS.md#explicit-capacity-extensions)
for `--context-limit`, `--max-candidates`, native runtime patching and extrapolation
metadata. Use fresh run IDs, smoke-test each model/configuration, preserve all
original measurements, and validate complete coverage before reporting totals.

### Liquid decision models

Configure Liquid credentials in the local `.env` using [`.env.sample`](../.env.sample).
From `evaluator/`, run a smoke check before evaluating the complete category:

```sh
uv run --env-file ../.env s1mb run --adapter liquid --model d1:free \
  --category smoke-v1 --limit 2 --run-id liquid-d1-smoke-001
uv run s1mb validate data/results/liquid-d1-smoke-001
uv run --env-file ../.env s1mb run --adapter liquid --model d1:free \
  --category english-v1 --run-id liquid-d1-full-001
uv run s1mb validate data/results/liquid-d1-full-001
```

The adapter preserves structured questions and authored Noul criteria, maps Score
levels in ascending numeric order, and records the model identity returned by the
API. An alias returned unchanged does not identify an immutable model revision.
Inputs are sent without local truncation. Probability distributions are normalized
only within four-decimal rounding tolerance. Use fresh run IDs for subsequent runs.
Use `--case-batch-size 8` for up to eight concurrent independent requests; the
default is one and the maximum is 32. Smoke-test the chosen concurrency first.

For additional public dedicated checkpoints, see the adapter contracts in
`evaluator/OPEN_MODELS.md`. Candidate-code and context extensions must be explicit
CLI options and remain visible in the saved model metadata. Run one measured GPU
job at a time. Preserve failed attempts under their original run IDs, diagnose
failures from local logs, and use a fresh run ID after each correction. A saved
partial run passing structural validation is still incomplete coverage.
