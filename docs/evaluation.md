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
The `llama-cpp` adapter connects to a caller-managed native `/v1/systemone`
server. Configure its URL, concurrency and declared runtime metadata through
`--adapter-kwargs` JSON; see [llama.cpp setup and limits](../evaluator/OPEN_MODELS.md#caller-managed-llamacpp).
It never starts or stops the server.

When a model can run through a llama.cpp API server, first consider whether it
can be evaluated with the shared `llama-cpp` adapter before adding a
model-specific adapter. Confirm native `/v1/systemone` support, preserved input
semantics, and overflow rejection; support for chat completions alone is not
sufficient.

Winnow also requires a separately built CUDA server; its `--server-host` must be
the machine's Tailscale IPv4 address or localhost when unavailable, and
`--server-port` must be unused. APUS needs its own pinned Transformers environment.
The `torchcast` and `solomon` bridges load the pinned author's Python runtime
through `--source`. Torchcast retains its native wide-choice tournament and
original task calibration. Solomon retains question-side LoRA, its trained
semantic heads and serving identity checks; it supports at most eight candidates.
Use a dedicated environment matching the selected release's requirements, and
verify GPU parity before smoke checks. Do not change dependencies used by waiting
or running evaluations.

The `jev27`, `eikos-fp8` and `jade` bridges connect to an already verified native
vLLM backend with `--server-host` and `--server-port`; they also require pinned
release code through `--source`. Keep the released converted adapters or FP8
quantization, full candidate log probabilities, calibration and tokenizer
identity. Bind local backends to Tailscale IPv4 or localhost and select the
physical GPU explicitly. JADE retains its hard 8,192-token limit including the
answer token; its requested candidate probabilities are read in chunks of 128.
Larger inputs remain visible errors. See the external adapter notes for each
model's exact conditions before starting a fresh run.

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
MetaEncoder-think uses the same extra and a local frozen reasoning-context file;
generate and protect that intermediate as described in
[its adapter notes](../evaluator/OPEN_MODELS.md#metaencoder-think).

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

### Unee

Unee is self-hosted: `unee serve` (from `pip install unee`) starts llama.cpp with a
Unee GGUF file and serves the same typed-question contract as TypeSafe, with no
API key. Start the server, then run the adapter from `evaluator/`:

```sh
pip install unee huggingface_hub
hf download uneeverse/unee-0.8b-GGUF unee-0.8b-Q4_K_M.gguf --local-dir .
unee serve --model unee-0.8b-Q4_K_M.gguf --model-name unee-0.8b-Q4_K_M \
  --gpu-layers 99 --port 8000
uv run s1mb run --adapter unee --model unee-0.8b \
  --category smoke-v1 --limit 2 --run-id unee-08b-smoke-001
```

Set `UNEE_BASE_URL` when the server is not on `http://127.0.0.1:8000`. The saved
revision is the server's `--model-name`. The server rejects inputs longer than a
slot; run long-document benchmarks with a slot that fits them, for example
`--slots 1 --ctx 32768`, and disclose it.

For additional public dedicated checkpoints, see the adapter contracts in
`evaluator/OPEN_MODELS.md`. Candidate-code and context extensions must be explicit
CLI options and remain visible in the saved model metadata. Run one measured GPU
job at a time. Preserve failed attempts under their original run IDs, diagnose
failures from local logs, and use a fresh run ID after each correction. A saved
partial run passing structural validation is still incomplete coverage.

## Additional small encoder models

Use the `julia` and `dinah` adapters with an explicit model revision and CUDA
device. Both accept `--context-limit` only within their native capacity and reject
input overflow. See [runtime notes and smoke commands](../evaluator/OPEN_MODELS.md#julia-and-dinah)
and the [model preparation queue](model_expansion.md). CPU input preflight is
useful before GPU smoke checks: model availability and small parameter counts do
not guarantee complete benchmark coverage.


The separate `jevk5-lite` adapter requires a pinned native source checkout and
explicit CUDA. It checks complete input tokenization against the native
512-token budget before inference and retains the native single-label calibration.
See [JevK5-Lite runtime and smoke instructions](../evaluator/OPEN_MODELS.md#jevk5-lite).


The `lavoir` adapter evaluates slot-free decisions with the native general-data
calibration, preserving full fields and rejecting input/head overflow. Use a
pinned native checkout, explicit CUDA and fresh run IDs. See
[Lavoir runtime and smoke instructions](../evaluator/OPEN_MODELS.md#lavoir).


`lfm-rlcd` supplies a local bounded bridge for LFM RLCD releases. It preserves
native full JSON-value likelihoods, explicitly reports an uncalibrated
likelihood softmax, and checks complete input paths without truncation. See
[LFM RLCD runtime requirements](../evaluator/OPEN_MODELS.md#lfm-rlcd). GPU smoke
and native-score parity checks precede full evaluation.

### Decision 2.0 native package

Use `--adapter decision2 --source /path/to/pinned/snapshot` with the exact
checkpoint revision. The bridge loads the release's verified native package,
retains its calibration and Score bias, and uses SDPA with native BF16 exact
linear residency and FP32 heads. It sends one anonymous question per call,
preserves structured criteria, authored Noul definitions and numeric Score
levels, and decodes probabilities in declared option order. Native Score supports
two through ten levels. Overflow is rejected without truncation; `--context-limit`
may lower the release's manifest budget but cannot raise it.

Validate a fresh three-task smoke before a full category run. Graphs, fused
kernels and shared-context optimization are disabled for this evaluation bridge.
Source digests, native manifest identity, base provenance, calibration and
resolved checkpoint revision are recorded in model metadata.

### OneJev native runtime

Use `--adapter onejev --source /path/to/pinned/OneJev/source` with an exact
checkpoint revision and the `open-models` and `onejev` extras. The native
multimodal engine receives text-only benchmark requests with no media. One
anonymous question is sent per call with complete structured state, authored
Noul criteria and numeric Score values. The native renderer, slot readout and
option ordering remain intact. BF16 model weights and the native FP32 readout
head use SDPA; sequential cache branches avoid batch-dependent rounding.

The default complete branch budget is 32,768 tokens, with overflow rejected
without truncation. Calibration is loaded only from the pinned checkpoint's
`calibration.json`; if absent, native temperature 1 is recorded explicitly.
Native six-decimal probabilities are normalized only within their rounding
bound. Validate GPU output parity and a fresh three-task smoke before a full
category run. CUDA graphs and media loading are disabled.

### Jev-Style v3 native runtime

Use `--adapter jev-style --source /path/to/pinned/snapshot` with an exact model
revision. The bridge verifies the native release manifest and retains its
renderer, FP32 Yes-minus-No readout, global calibration temperature and lossless
option chunking. It sends complete structured criteria, anonymous Choice names,
authored Noul definitions and numeric Score levels in declared order. Structured
instructions are serialized as JSON to the native string instruction field.

The native default budgets are 25,600 total tokens and 2,048 head tokens.
Choice candidate descriptions exceeding the head budget use native contiguous
option chunks; the complete per-option scores share one calibrated softmax.
Other overflow is rejected without truncation. Use GPU SDPA, with CUDA graphs
disabled. Verify all three tasks and a large split-option Choice against native
probabilities before a fresh validated smoke and full category run.

### Sifr native option-key scorer

Use `--adapter sifr --source /path/to/pinned/decision-index/source` with the exact
Sifr checkpoint revision and open-model plus vision dependencies. The snapshot's
`sifr_engine.py` provides the native renderer and mean option-key log-likelihood
softmax. Both dependency-source and checkpoint-source digests are recorded.
The released identity temperature is verified. BF16 GPU inference uses SDPA,
one question per call and one full-sequence branch per batch.

The native Noul renderer ignores authored definitions and the native API does
not implement Score. The bridge therefore sends every primitive as an explicit
Choice distribution, preserving structured instructions, anonymous Choice
labels, authored true/false definitions and numeric Score level descriptions.
It maps probabilities back to the original declared option order. This typed
mapping is recorded in metadata.

The default complete input budget is 32,768 tokens; overflow is rejected by the
native scorer without truncation. `SIFR_KV=1` is refused because this bridge
uses full-sequence native scoring. Verify typed probability mappings on GPU and
a fresh three-task smoke before a full run.

### Jiwo native runtime

Use Python 3.12 for the pinned native jiwo source (it uses Python 3.12 type
alias syntax), with `--adapter jiwo --source /path/to/pinned/jiwo/source`.
Keep this evaluation environment separate from the regular Python 3.11
workspace. The bridge loads pinned full decoder and readout weights, uses GPU
BF16 SDPA, and preserves native per-type temperatures and option alignment.
Structured instructions, authored Noul definitions and numeric Score levels
are passed without targets or stable benchmark identifiers.

The bridge explicitly uses a 32,768-token native maximum and batch budget,
with one anonymous question per call. Native overflow validation occurs
before computation and never truncates. Record actual Torch, Transformers
and Python versions; verify native probabilities and a fresh three-task smoke
in the isolated environment before a full category run.

### JPT with the llm2jev native HF reference

Use `--adapter llm2jev --source /path/to/pinned/llm2jev/source` and an explicit
`--temperature`; JPT-0.8B's pinned release declares `--temperature 1.140`.
The bridge retains llm2jev's native chat renderer, thinking-disabled template,
context-verified 255 single-token labels and full-vocabulary HF reference
readout. It sends one anonymous question at a time with complete structured
criteria, authored Noul definitions and numeric Score level descriptions.

GPU inference uses BF16 SDPA. Complete tokenization is checked against the
default 32,768-token budget before native scoring; no truncation or CPU
fallback is allowed. Role-bearing records with missing chat fields or extra
metadata are serialized intact as JSON instead of losing fields in the native
chat renderer. Text-only runs reject media. Validate native typed probabilities
and a fresh three-task smoke before running the full category.

### Intern-Decision native masked-symbol runtime

Use Python 3.12 or newer with `--adapter intern-decision` and
`--source /path/to/pinned/snapshot`. The bridge loads the snapshot's native
`inference.py`, keeps its trained decision marker and immediate-predecessor
logit readout, and uses the release's own temperature. GPU BF16 SDPA processes
one anonymous typed question per call. Complete structured criteria, authored
Noul definitions and numeric Score levels remain in declared order.

The explicit default token budget is 32,768, validated against checkpoint
capacity; native length checks reject overflow without truncation. The released
runtime supports at most 62 candidate symbols and rejects larger questions
without filtering or replacing candidates. Reserved decision markers in input
are rejected by the native compiler. Keep any such partial results visibly
incomplete and audit the failures against the pinned native compiler. Verify
typed GPU probability parity and a fresh validated smoke before a full run.

For `flock-io/this-that-model-1.2`, `--adapter thisthat --source <pinned-checkout>`
uses the native PyTorch answer-slot backend on explicit indexed CUDA devices.
The bridge sets a full-state token budget and rejects complete padded prompts
above 32,768 tokens; it never applies the native default state truncation.
Structured inputs and numeric Score levels remain visible in native question
text. The native constructor rejects identical option descriptions. Run GPU
parity and the three-task smoke, validate its saved results, then start a fresh
full run; CPU input audits alone do not establish inference support.

EXAONE-JEV uses `--adapter exaone-jev --source <pinned-checkout>` with the
`exaone-jev` optional extra. The adapter retains native per-type temperatures,
two option orders and multi-round large-choice scoring, while reading label
logprobs with Transformers. Its explicit 32,768-token limit rejects overflow.
Check GPU parity for Choice, Noul, nonuniform numeric Score levels and a wide
choice before the three-task smoke/full sequence. Every wide-choice candidate
must remain represented after native final-round mass redistribution.

Jev-Style 2B v3 uses `--adapter jev-style-2b` with the pinned snapshot as
`--source`. Retain the native block-causal SDPA backend, FP32 and global
calibration; the 0.8B runtime's causal attention and option-chunk configuration
are not interchangeable with this release. Include an overflowing short-form
catalogue in the GPU parity check, confirm all candidates survive, and validate
smoke results before starting the full 137-benchmark run.

Sieve requires `--adapter sieve --source <pinned-snapshot> --base-revision <sha>`.
The dependency checkpoint SHA is explicit and saved in model metadata. Native
FP32 prefix-fork scoring keeps every candidate and the released temperature.
Overflow is rejected before native trimming; the full declared input limit can
only be lowered through `--context-limit`. Check native probability and option
permutation parity, then validate three-task smoke results before the full run.
Native duplicate option descriptions are recorded as failures, without adding
invented differences or fitting a schema to benchmark targets.

For the LFM2.5 2.6B RLCD release use `--adapter lfm-pcd --source <pinned-snapshot>`
and install the `lfm-pcd` extra. Its native PCD token mode differs from the 350M
sequence-likelihood bridge. Explicit configured limits are recorded; complete
prefix and field suffix must fit without truncation. Retain native conservative
GPU memory checks. Require cache validation and cached-versus-uncached full-head
probability parity before the three-task smoke, then validate smoke/full results.

Kodiak accuracy mode uses `--adapter kodiak --source <pinned-checkout>` and the
`kodiak` optional extra. All three ensemble members count toward parameter
metadata. Native schema/token budgets are checked without truncation. Forced
Choice/Noul requests explicitly disable abstention. Numeric Score distributions
are the native moment-matched Beta integrated between numeric-level midpoints;
all authored descriptions remain in native Score text/anchors. Require native
public-output parity and validated smoke before full evaluation. Expect declared
schema/token capacity failures to remain visible in saved partial results.

Lev1 3.7B uses `--adapter lev1 --source <pinned-snapshot>`. It retains native
packed block-mask SDPA, cached long-input scoring and original/reversed option
order averaging. The default complete-input limit is 32,768 tokens, without
truncation. Authored Noul definitions and numeric Score levels use explicit
native Choice criteria. Require native parity including cached long-option
scoring, then validate three-task smoke results before full evaluation.

Kas 4B uses `--adapter kas --source <pinned-kas-checkout>` and the `kas`
optional extra, which pins the native engine protocol dependency. Native BF16
SDPA, repeated JSON prompting, prefix-cache reuse and restricted-label softmax
at temperature 1 remain unchanged. Every task uses explicit Choice criteria
to preserve authored Noul definitions and numeric Score levels. The default
complete prompt limit is 32,768 tokens; excess candidate counts and context
overflow are rejected without truncation. Require native parity and validated
three-task smoke before full evaluation.

Intelif uses `--adapter intelif --source <pinned-intelif-checkout>` in Python
3.12 or newer. Its release config pins the external base checkpoint revision.
The native loader merges released LoRA and uses its linear anchor readout with
BF16/native SDPA GQA. No ordinary HF or PEFT-only inference path is substituted.
The native 40,960-token default limit may only be lowered. Explicit Choice
transport retains authored Noul definitions and numeric Score levels. Native
GPU parity and validated smoke must pass before full evaluation.

The current Lev release uses `--adapter lev --source <checkout>/packages/lev`
and requires `--base-revision <exact-sha>`. Run in Python 3.12 or newer. The bridge
loads native `mode_b_head.pt`, merges released LoRA into the pinned base and uses
native calibration and routing with BF16/SDPA. Explicit Choice transport keeps
authored Noul definitions and numeric Score levels. Reject complete-input
overflow and Mode B candidate text longer than the native 64-token budget before
forward, rather than accepting native truncation. Clear candidate caches between
questions. GPU parity and validated smoke remain required before full evaluation.

Jet uses `--adapter jet --source <pinned-release-snapshot>` with Python 3.12
and its released Torch 2.11/FLA runtime. Retain native BF16 SDPA, FLA gated-delta
scoring and PyTorch convolution. The native complete-input limit is 16,384
tokens and may only be lowered. Explicit Choice transport retains authored
Noul and numeric Score definitions as lossless strings, with native Choice
temperature. Require GPU parity and validated smoke before full evaluation.

Hopper uses a pinned native checkout with Python 3.12 or newer. Retain native
BF16/SDPA, exact base revision, calibration, and constructor kernel checks.
Disable shortlisting and reject more than 26 candidates rather than removing
options or extending the native head. A declared 32,768-token evaluation budget
is checked without truncation. Record native kernel/fallback diagnostics; do not
replace kernels after their native validation. Require native GPU parity and
validated smoke before full traversal, and audit recorded capacity failures.

Pngwn requires `--base-revision <exact-sha>` for its external backbone. The
release temperature is read from its pinned `metrics.json`, validated and
recorded; do not reuse another checkpoint's temperature. Preserve all candidates,
full state/question/option text and numeric Score values. Extended full-input
evaluation rejects overflow instead of native training/demo truncation, and
records that condition explicitly. Verify readout/probability parity and smoke
results against this configuration before full traversal.

RSI uses `--adapter rsi --source <pinned-rsi-checkout>`. The native in-process
server loads the self-contained release and its calibrated option readout.
Explicitly set the complete-input budget (default 32,768) and `truncate=none`;
verify the native encoder activated those settings. Use explicit Choice criteria
for authored Noul definitions and numeric Score levels. Preserve the trained
option pooling and every candidate, reject any truncation report, and require
native GPU parity plus validated smoke before full traversal.


For the Xor adapter, prepare the release-pinned SGLang image and verify the model
checksum manifest before loading the snapshot at `/models/xor`. Select physical
GPU 1 on the shared workspace and bind the backend to the resolved Tailscale IPv4
address or localhost. Pass `--server-host` and `--server-port` to `s1mb run`.
Flush its cache before each new run; stop the owned backend when validation ends.
The adapter uses the release's compatibility-server rendering and calibration,
and an alternative inference runtime requires independent validation.


Blink's NVFP4 backend requires vLLM's selected-token logprob API, `modelopt_fp4`
quantization and FP8 KV cache. Use the verified model snapshot at `/models/blink`,
served as `blink`, and pass its safe bind address/port to `s1mb run --adapter blink`.
The backend must return all requested token IDs; do not replace missing candidate
probabilities with guessed logits. Verify complete native prompt tokenization and
reset the prefix cache before independent parity comparisons and fresh runs.
Use `VLLM_BATCH_INVARIANT=1` for the verified Blink runtime and confirm repeated
identical requests return identical candidate log probabilities before evaluation.
The adapter requests candidate token IDs in chunks of at most 128, then applies
calibration and normalization once over all candidates.

For JEV-27B, vLLM 0.31 rejects batch-invariant mode with GDN attention. The verified
configuration uses eager execution, disables prefix caching, selects the Triton
GDN prefill backend through `--additional-config '{"gdn_prefill_backend":"triton"}'`,
and sets the LoRA shrink kernel's `split_k` to 1 through
`VLLM_TUNED_CONFIG_FOLDER`. Use the runtime's GPU-specific tuned-config filename
and schema, preserve other kernel parameters, and record the configuration checksum.
Parallel split-K accumulation can vary between identical requests; disabling CUDA
graphs and prefix caching alone does not remove that variation. Require repeated
request stability, native parity including wide candidates, and validated smoke
checks before a full run. Preserve the released weights, precision and calibration.
