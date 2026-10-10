# External model adapters

S1MB keeps each external inference implementation in a separate checkout.
`upstream-models.json` lists source repositories and known checkpoint references;
these are configuration references, not benchmark results or endorsements.
Install the dependencies required by the selected upstream model separately.
No upstream code or weights are vendored into the evaluator.

Use `--source /path/to/checkout`, an explicit `--model` and `--device`, and run
`smoke-v1 --limit 2` before a complete evaluation. Record distinct configurations
under distinct run IDs. Inspect the result's model metadata for resolved weights,
source hashes, rendering, attention backend, precision and input limits.

| Adapter | Interface |
| --- | --- |
| Bekko v0 (`bekko-v0`) | Hub remote `BekkoSentenceTransformer.predict()`; pinned code and weights, no local checkout |
| System Ichi | Local typed model with bounded question batching |
| Laya | Full native token layout, FP32 arithmetic, bounded cross-case batches; no CPU fallback |
| Meta Encoder (`meta-encoder`) | Independent text-choice embeddings with joint cosine softmax |
| Von | Native option-marker backend and calibration |
| JevForge | Candidate-path inference followed by a joint softmax |
| Kev | Native typed loader with explicit input budgets |
| Decider | Native typed API with independent questions |
| JevK5 | Native calibrated letter logits and candidate selection |
| Minojev | Native decision head and calibrated candidate scores |
| Open-Jev | Zefan Cai's pinned Qwen backbone, LoRA, scalar head and saved temperature |
| OpenJev 27B (openjev organization) | Native text prompt, exact letter logits and fixed calibration |
| Alex Openjev | Alex Wortega's NLI head, native overlapping windows and normalized entailment |
| CLM | Native typed rendering and contrastive heads with local last-token Qwen3 embeddings |
| Tev | Native decision JSON and A–X letter probabilities, with an explicit large-menu extension |
| Luce | Standalone inference with recurrent backbone and typed criteria |
| Verdict2 | Requires a compatible `model.pt` checkpoint |
| Openvons | Requires a trained text checkpoint with `head.pt` |
| Lumma (`lumma`) | Released native backbone, decision head and rounded typed readout |
| TinyJev (`tinyjev`) | Native Torch backbone and NumPy pointer head |
| Nimble (`nimble`) | Packaged schema rendering, hashes and candidate logits |
| Bosun (`bosun`) | Stable candidate slots, trained decision embeddings and LoRA |
| Manchego (`manchego`) | Native short prompt and extended contract-v2 codebook |
| NeoHorse (`neohorse`) | Native branch engine, one question per call |
| Jebadiah (`jebadiah`) | LoRA or merged BF16 checkpoints, native AINode renderer and saved per-task temperatures |
| OpenThai (`openthai`) | Native typed client with one candidate permutation |
| Kotoba (`kotoba`) | Released DeBERTa backbone plus trained span-pooling head |
| Jeff (`jeff`) | Native prompt and automatic first-token or whole-label scoring |
| Pngwn (`pngwn`) | Trained scalar option head and native split tokenization |
| Mini-Jev (`mini-jev`) | Released NF4 backbone and joint candidate head |
| Eikos (`eikos`) | Native SemIf prompt, candidate letters and temperature |
| Openjev shim (`openjev-shim`) | Native calibrated shim with local exact candidate logits |
| Verdict encoder (`verdict-encoder`) | Native GLiClass labels and calibration, conditioned on non-abstention |
| APUS (`apus`) | Native high-effort choice probabilities with an explicit enum bridge |
| Winnow (`winnow`) | Independently built native CUDA server and verified Q8 GGUF |

Support for an adapter does not imply that a checkpoint is publicly available or
that every benchmark fits its context limit. Failed or partial runs remain visibly
incomplete. Do not interpret arbitrary upstream truncation or unavailable weights
as successful evaluation. Model and source licenses must be reviewed separately.

## Meta Encoder

`meta-encoder` evaluates `facebook/meta-encoder` as an independent text-choice
encoder. Install its runtime with `uv sync --extra meta-encoder`; install a
compatible FlashAttention build separately when selecting
`--attention flash_attention_2`. The adapter resolves the requested Hub revision,
loads Muse Glimmer in BF16, left-pads text-only batches, pools the final token from
the last hidden layer, converts embeddings to normalized FP32, and applies one
joint cosine softmax over each decision's declared candidates.

The renderer prefixes each query with `Select the correct option.`, retains the
raw structured state JSON, and includes the dataset instruction, criteria, and
option IDs in declared order. Candidates are the option IDs encoded independently.
The explicit positive `--temperature` is part of the measured configuration and
is saved in every result; the adapter has no implicit inference temperature.

The default context limit is 8,192 tokens. Inputs are tokenized intact and
overflow is rejected before model inference. An explicit larger
`--context-limit`, within the checkpoint's supported position capacity, evaluates
full longer inputs and is recorded as an evaluation condition. Candidate encoding
is bounded to eight texts per batch and reuses at most 4,096 exact candidate
embeddings. Targets, annotations, case identifiers, and provenance are never
included in model text.

Example smoke command:

```sh
CUDA_VISIBLE_DEVICES=1 uv run s1mb run \
  --adapter meta-encoder --model facebook/meta-encoder \
  --revision cb36037ca0e6276c92fed2d7a2a9fbcdd4e79109 \
  --temperature 0.03 --device cuda:0 --attention flash_attention_2 \
  --context-limit 8192 --category smoke-v1 --limit 2 \
  --run-id meta-encoder-smoke-001
```

## Additional English model runtimes

The typed-decision ecosystem directory tracks model families, candidate entries,
and implementations with different readiness levels. Being listed there does not
establish runnable weights, English coverage, or support for all three S1MB tasks.
Use the pinned references in `upstream-models.json` and inspect each release's
model card and runtime. XERON uses the existing Laya adapter; Decider 2B uses the
existing Decider adapter.

The additional bridges use bounded, independent questions and preserve the
dataset's authored instructions. Choice IDs are anonymous. Native Noul criteria
are retained where supported. Kotoba, Nimble, APUS and Verdict use explicit enum
representations to retain authored Noul descriptions and numeric Score levels
when their convenience interfaces cannot express these values. This is a recorded
evaluation condition, not a claim that APUS's binary `score_level` is S1MB Score.
Mini-Jev receives all state as its summary field, preventing its tool-state
whitelist from dropping arbitrary S1MB evidence.

Lumma, TinyJev and Mini-Jev retain their native input limits and reject overflow
before upstream truncation. Kotoba retains both its state and total-token limits.
Nimble supports up to 255 candidates and 8192 tokens. APUS retains its 16-candidate,
8192-token limits. Verdict admits up to 24 substantive candidates and 8192 tokens,
an extended-input condition compared with its runtime's 512-token default. Its
native abstention probability is removed by conditioning on the supplied S1MB
candidates; this is recorded in model metadata.

Manchego, NeoHorse, Jebadiah, OpenThai, Jeff, Pngwn and Eikos admit full inputs up
to 32768 tokens, recording where this extends native defaults or training
conditions. Pngwn retains the demo's separate state/suffix tokenization and
temperature 1.75, but removes its field slicing and 32-option cap. OpenThai uses
one declared-order permutation, and its native readout conditions on the supplied
candidates. Jeff uses a recorded codebook of anonymous uppercase Choice labels
with distinct native leading-space tokens. It retains automatic whole-label
scoring when labels share their first token, including numeric Score labels.
Openjev's shim preserves its native composition above 52 candidates,
choice temperature 0.85 and Noul temperature 1.829074 with zero bias. Its local
transport returns exact candidate logits instead of top-vocabulary approximations.

Install `.[mini-jev]` for Mini-Jev's bitsandbytes dependency. Verdict additionally
requires the upstream `gliclass` package. APUS requires its audited
`transformers==5.16.1`; use a separate environment instead of modifying an active
model evaluation environment. Eikos and Openjev distribute contiguous decoder
layers over the explicitly visible GPUs, reserving room for the output head and
rejecting CPU placement. Other bridges use one GPU. Openjev's upstream helper
requires the `openai` package to import; the adapter replaces its HTTP transport
with local inference.
Install NeoHorse's matching release wheel with `pip install --no-deps
dist/neohorse_decision-1.0.0-py3-none-any.whl`; its runtime reads installed package
metadata even when `--source` points directly to the source tree.

Winnow requires building the pinned checkout with `scripts/build.py` and CUDA.
Install `uv sync --extra winnow` for GGUF parameter counting. Counts use logical
tensor shapes rather than quantized storage sizes. Shared token/output embeddings
are excluded from static AP under `embedding_excluded_parameters_v1`. The verified Q8 release
(`b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea`)
contains 11,907,350,576 total parameters across 667 tensors; static AP
subtracts its token embedding table.
Its adapter verifies the GGUF against the checkout's release manifest, starts the
native server, and closes it after evaluation. Use `--server-host` with the
machine's Tailscale IPv4 address, or localhost if unavailable, and an unused
`--server-port` (default 8091). Expose exactly one GPU explicitly. The recorded
condition is Q8 weights and KV cache, selected output head, 65536 context positions,
one decision branch, and native prefix reuse. Its 64-candidate bound is retained.
No native source, model weights, local audit reports or measurements belong in
the source repository.

## CLM and Tev

CLM uses the published projection heads and a pinned Qwen3-8B encoder. State and
action inputs are encoded separately with last-token pooling and L2 normalization;
the native FP32 heads and saved logit scale produce the typed distributions.
The local SDPA implementation retains full inputs up to the configured 32768-token
limit, rejecting overflow. This is a full-input condition rather than the official
HTTP example's 2048-token limit. State embeddings are computed on every call.
Action projections use the official bounded GPU cache with a fixed 64 MiB budget,
keyed by exact candidate text and head identity. Evaluation time includes cache
misses and subsequent reuse; report this condition when comparing throughput.

Tev retains its native system prompt, decision JSON and thinking-disabled chat
template. The adapter takes a temperature-1 softmax over the allowed A–X token
logits at the first answer position. These are conditional label probabilities,
not a claimed calibrated probability API; official examples return a single label.
Only the final token passes through the vocabulary head (`logits_to_keep=1`).
For more than 24 candidates, S1MB uses balanced contiguous groups in declaration
order and multiplies group and within-group probabilities. All candidate text and
probability mass remain present, but this is an explicit S1MB extension to the
official 2–24-option interface. Group composition can affect the predictions.
Both adapters use `--case-batch-size 1` and reject longer inputs without truncation.

Primary sources: [CLM model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B),
[CLM source](https://github.com/Contrastive-LM/CLM),
[Tev model card](https://huggingface.co/togethercomputer/Tev1-4B-experimental), and
[Tev source](https://github.com/togethercomputer/tev1).

## Input fidelity and throughput

For Bekko Hub models, see [Bekko setup](README.md#bekko).
The adapter preserves native rendering and candidate order, delegates input
budgeting and truncation to the current native runtime, and records the policy
and exported code/weight hashes. Native attention uses dense masks and gathers shared
prefix keys/values for candidate paths; padding and long contexts can therefore
dominate despite a linear token budget. Benchmark batching and optional compilation
on the intended hardware before selecting throughput settings.

Except for Bekko v0's documented native truncation, local adapters reject inputs
exceeding their explicit budgets. The full-input Laya
renderer retains instructions, candidate descriptions and state in the native token
layout; it removes the upstream renderer's 48-token candidate cap and field slicing.
This is an extended-input condition when it exceeds the checkpoint's training
length. Laya uses FP32 arithmetic to reduce batch-dependent probability drift.
Its forward path calls the GPU model directly, without the upstream API's CPU
fallback. `--max-len` specifies total input admission and `--head-max-len` bounds the
question plus candidates. Both limits reject overflow instead of truncating.

Von preserves independent-option attention masks and input-dependent calibration.
Kev preserves its native head temperature. Minojev batches backbone paths while
retaining the decision head's joint candidate attention. JevK5 batches small
questions and retains native knockout selection above 16 candidates. The Qwen3.5
adapters enable supported Transformers Hub kernels for recurrent layers; SDPA and
these recurrent kernels serve different operations.
With the pinned Torch 2.10 runtime, convolution stays on the native GPU path;
the Hub convolution layer builds require a newer Torch version. Decider and JevK5 process
one case at a time because their BF16 cross-case batch probes exceeded the accepted
probability drift, while retaining native candidate processing within each case.
Kev supports `--case-batch-size`; the audited 9B condition uses 1 after its
16-case BF16 probe exceeded the probability-drift threshold. The 0.8B and 4B
conditions retain 16 cases per call.
Open-Jev likewise supports `--case-batch-size`; its audited 9B condition uses 1,
while 2B retains 16. Both preserve native within-case candidate order.
Minojev's FP32 checkpoint path is the measured default: converting its backbone
to BF16 produced substantial batch-dependent drift in the preflight probe.
Its FP32 SDPA bridge expands grouped keys and values explicitly and requires the
GPU memory-efficient backend, avoiding the quadratic-memory math path selected
by implicit FP32 GQA. Identical state prefixes of at least 512 tokens shared by
four or more candidate paths are cached on GPU, with fresh bounded candidate
caches. Suffix positions, full causal context, candidate order and the native
joint decision head remain intact. Long-input native/cached parity is checked
before measurement; no context is discarded to fit memory.

Choice identifiers rendered by upstream APIs are replaced with positional labels
and mapped back after inference. Noul retains authored true/false descriptions;
Score distributions map back to the declared option order and numeric values.
Batching groups independent rows under padded-token budgets. It never drops a
candidate or splits the final per-question probability normalization.

For timing, load the model once, warm up all three task types, verify native and
batched predictions, synchronize CUDA around measured calls, and run one model
at a time on GPU 1. Report model loading separately from evaluation. Saved
`elapsed_seconds` includes input rendering, tokenization, inference, probability
validation, metadata and metric calculation, but excludes dataset loading and
result serialization. It is throughput evaluation time, not isolated request
latency. Store timing diagnostics outside result roots.

## Other Open-Jev implementations

[Zefan Cai's Open-Jev](https://github.com/Zefan-Cai/Open-Jev) uses a LoRA adapter,
a scalar head and a saved calibration temperature. The model loader resolves the
base model revision recorded in `package/checkpoint/model.json`. Use adapter
`open-jev`, the upstream source checkout, and model `ZefanCai/Open-Jev-2B` or
`ZefanCai/Open-Jev-9B`. Independent candidate batches retain the native chat
rendering. Prefix caching is disabled. An explicit `--context-limit` beyond the
published 4,096-token limit is an extended-input evaluation condition.

[Alex Wortega's Openjev](https://huggingface.co/AlexWortega/openjev) uses three-way
NLI probabilities. Use adapter `alex-openjev`, model `AlexWortega/openjev`, and
`--subfolder qwen3.5-2b-nli-v5` or `--subfolder qwen3.5-4b-nli-v5`. `--source`
points to a local checkout of the model repository's Python source. The adapter
retains the published typed-decision hypothesis template, 24,000-character
windows with 2,000-character overlap, maximum entailment across windows and
normalization across candidates. Every encoded pair is checked before inference;
the upstream pair encoder's silent truncation is removed. Windowing preserves
text coverage but does not preserve all cross-window relationships.
Inference preserves the native per-question call boundary and candidate order:
combining questions changed BF16 probabilities beyond the preflight tolerance.

The Alex Wortega v5 model card explicitly discloses training on test splits of
several public benchmarks. Consult its `panel_manifest.json` and model card when
interpreting overlapping evaluations; these scores do not establish unseen-task
generalization. Model and source licenses remain separate from S1MB's license.

## OpenJev 27B from the openjev organization

`openjev-org` evaluates `openjev/openjev`. This is a different model and interface
from `open-jev` and `alex-openjev`. Use an unambiguous display name such as
**OpenJev 27B (openjev/openjev, BF16)** and model folder
`openjev__openjev_27b_bf16`; never merge its results into another Open-Jev row.

Install the `open-models` extra. Pin the model to an exact Hub SHA and point
`--source` at that snapshot's `helper/` directory to record the reference source
hash. On the shared workspace, select physical GPU 1 as described in the
[evaluation guide](../docs/evaluation.md), then run:

```sh
uv run --extra open-models s1mb run \
  --adapter openjev-org --model openjev/openjev --revision MODEL_SHA \
  --source /path/to/pinned/snapshot/helper --device cuda:0 \
  --category smoke-v1 --limit 2 --run-id openjev-org-27b-smoke-001
```

The adapter uses Transformers BF16 and SDPA, with supported hybrid attention
kernels. This is a text-only evaluation of the complete checkpoint. It is not the
model card's vLLM on-load FP8 serving condition. The native chat template disables
thinking, and one forward pass reads exact candidate-letter logits. Calibration
is fixed to the published settings: temperature 0.85, Noul temperature 1.829074,
zero Noul bias, and the native boolean clipping interval. Probabilities retain
full precision instead of the HTTP helper's four-decimal output rounding.

Structured instructions use the published Python-representation convention;
structured states and descriptions use JSON. Choice keys are anonymous, Noul
retains authored positive/negative definitions, and Score options are ordered by
their numeric values for the native lowest-first menu, then mapped back to their
original IDs for scoring. No targets or provenance enter prompts.

Above 52 options, the adapter uses native near-equal chunks and a final readout
of chunk winners, anchoring each chunk's entire distribution on its winner.
This is an approximation, not a single global softmax. At most 2704 options are
supported. Each prompt must leave room for one readout token within the default
16384-token limit; overflow raises an error without truncation. The context limit
may be lowered with `--context-limit`. Inference runs one question at a time.

Weights and the upstream helper have separate licenses; see the model repository.
The source adapter does not redistribute either artifact.

## Explicit capacity extensions

The default limits above remain the released serving conditions. For a separate
full-input evaluation, `--context-limit` also supports TinyJev, Nimble, Lumma,
Mini-Jev, APUS, Verdict encoder and Kotoba. Use fresh run IDs and retain original
measurements. These options never truncate inputs or discard candidates.

| Adapter | Extended condition | Preserved computation |
| --- | --- | --- |
| TinyJev | Up to the backbone position limit (32K for the evaluated releases) | Native pointer renderer, head and calibration |
| Nimble | Up to the backbone position limit | Native candidate prompt and logits; the 8K training contract is recorded separately |
| Mini-Jev | Up to the backbone position limit (32K) | Summary tokens are inserted at the exact native summary boundary; suffix pooling and joint head are unchanged |
| Lumma | Explicit row/state limit up to 32K | Native delimiters, pointer head and RoPE frequencies |
| Verdict encoder | Explicit context up to 32K and `--max-candidates 255` | Full dynamic-label forward, original abstention conditioning and released calibration |
| APUS | Backbone-supported context and `--max-candidates 255` | Original A–P prefix, additional unique single-token uppercase codes, native full-head logits |
| Kotoba | Explicit context up to 32K | Full relative attention, authored spans and head; query blocks of 128 bound temporary attention memory |
| Winnow | `--max-candidates 255` with the patch below | Native single-token codebook order, selected output rows and joint candidate softmax |

Lumma's checkpoint position limit is 12,288, ModernBERT's is 8,192, and Kotoba's
published DeBERTa setting is 512. Larger requested contexts are explicit position
extrapolation, recorded in model metadata; weights and positional frequencies are
not retrained or rescaled. Completing inference does not establish long-context
accuracy. Verdict's larger menus also extrapolate beyond its calibration scope.
APUS and Winnow's additional letter slots exceed their released menu conditions.
Keep these conditions visible when comparing or publishing results.

Mini-Jev's extension applies to the summary-only state mapping used by this adapter.
Its native token boundaries and option masks match the original encoder exactly
for inputs within the original budget. The native GPU method is rebound with an
instance-local encoder; other loaded modules are not modified.

Kotoba uses relative-only positional embeddings. Its extension tiles queries while
retaining all keys and both content-to-position and position-to-content terms.
It does not split documents into independent windows or average window predictions.
Synthetic tests compare tiled and original attention with padding and sequences
beyond the configured position length. Actual long-input GPU smoke checks remain
necessary before a full run.

For Winnow, apply `upstream-patches/winnow-255-candidates.patch` at the pinned
inference checkout root before the first runtime build, then build with its CUDA
script before requesting more than 64 candidates. The patch raises codebook
enumeration capacity and the limit on selected output rows; token uniqueness and
native request validation remain active. The adapter records
a hash of the native sources in addition to Python source and runtime-lock hashes.
Use the Tailscale/localhost binding instructions above when starting its server.

Jebadiah merged releases such as `frontier-infra/jebadiah-9b-v2` load their own
weights, tokenizer and chat template without applying another adapter. Set
`--source` to the `scripts` directory from the same pinned model snapshot. The
released `temperatures.json` supplies all three task temperatures, including the
v2 Score calibration update. The existing 32K full-input condition and overflow
rejection also apply to merged releases; checkpoint format and base revision are
recorded in result metadata.
The same path supports `frontier-infra/jebadiah-27b`. Structured descriptions and
numeric Score levels are serialized as JSON strings through native Choice
rendering, preserving authored criterion order while applying each original
task's saved temperature. The native FP32 candidate readout is required, and
each label is checked at the actual prompt's continuation boundary.

`kev` also supports Kev 27B v2 full-weight checkpoints through the native
`kev.checkpoint.Checkpoint` loader. Use a pinned Kev source checkout, a case
batch size of 1 for long inputs on one GPU, and an explicit context limit within
the checkpoint's capacity. The saved pointer-head temperature remains active;
all-choice transport preserves authored criterion order and numeric Score values.

## Firelex Jeff (mstrasser checkpoints)

`firelex-jeff` evaluates `mstrasser/Jeff-Qwen3.5-0.8B`,
`mstrasser/Jeff-Qwen3.5-2B`, `mstrasser/Jeff-Gemma4-E2B`, and
`mstrasser/Jeff-Qwen3.5-0.8B-Chess` using the pinned
[firelex/jeff](https://github.com/firelex/jeff) runtime. This is independent of
`jeff`, the adapter for `GestaltLabs/Jeff-1`.

The upstream runtime requires Python 3.12. Use an isolated environment for it;
the evaluator's regular development checks still run with Python 3.11:

```sh
git clone https://github.com/firelex/jeff ../tmp/firelex-jeff
git -C ../tmp/firelex-jeff checkout f06788292874c21a5b5c41549ac220dd9e15da7f
UV_PROJECT_ENVIRONMENT=../tmp/firelex-jeff-venv uv sync --locked \
  --python 3.12 --extra firelex-jeff
CUDA_VISIBLE_DEVICES=1 ../tmp/firelex-jeff-venv/bin/s1mb run \
  --adapter firelex-jeff --source ../tmp/firelex-jeff \
  --model mstrasser/Jeff-Qwen3.5-0.8B \
  --revision 0f212b3e72acb4dde3f7da61e925d6ab7f819990 \
  --device cuda:0 --category smoke-v1 --limit 2 \
  --run-id firelex-jeff-smoke-001
```

Inference uses the native prompt, backbone and trained readout, checkpoint
calibration, BF16, SDPA, and one question at a time. Qwen's recurrent layers use
supported Hub kernels. State and authored criteria retain their structured
values; anonymous Choice keys omit source identifiers. Score descriptions
include their numeric levels in declared order. Native Noul false/true order is
mapped back to the authored option IDs.

The default context limit is 8192 tokens, with overflow rejected before forward
inference. Candidate limits come from `decision_config.json`: the pinned Qwen
v1.1 checkpoints declare 254, Gemma declares 26, and Chess declares 101. An
explicit `--context-limit` or `--max-candidates` (at most 255) can evaluate larger
inputs with the same native codebook and weights. Metadata records both trained
and effective limits and whether a capacity extension was used. Such extensions
do not establish long-context or long-list accuracy. Inputs and options are
never silently shortened or shortlisted.

The Qwen v1.1 release discloses MASSIVE and CLINC150 training-split use. The Chess
checkpoint is specialized for chess; its S1MB measurements describe performance
on this benchmark, not its chess strength or unseen-task generalization.

### Additional dedicated decision checkpoints

The following adapters use explicitly pinned Hub snapshots and source checkouts.
They are separate from similarly named existing adapters. Use `--device cuda:0`
with `CUDA_VISIBLE_DEVICES=1` on the shared workstation. Always run `smoke-v1`
before `english-v1`, and validate each saved run. Runtime compatibility and full
coverage must be established for each checkpoint; adapter availability is not a
claim that a checkpoint has completed evaluation.

GLiFormer Jeff needs a separate environment: install the `gliformer-jeff` extra
without the Transformers 5.17 extras (`open-models`, `firelex-jeff`, `bekko-v0`,
or `laya`). GLiFormer 0.1.2 imports a GLiNER helper absent from 0.2.24; the tested
combination is GLiNER 0.2.29 and Transformers 5.16.0, matching GLiNER's declared
Transformers upper bound. These dependency versions are recorded in model metadata.

| Adapter | Native runtime / checkpoint family | Input and probability contract |
| --- | --- | --- |
| `jevlite` | `mghafiri/qwen3.5-0.8B-decision-model`, bundled `jevlite` source | Native calibrated answer logits; strict full-token length checks |
| `certo` | `altslate/certo-decision-model`, AltSlate-Labs/certo | Native independent option logits, calibrated across all options; no state or option truncation |
| `jev-omni` | `akhilaaa3/Jev-Omni`, bundled `jev_omni.py` | Native 256-way head; distinct anonymous prefixes preserve duplicate descriptions |
| `autojev` | AutoJev and `perplexity-ai/pplx-decider-v1.1-27b`, bundled `source/src/autojev` | Native decision head, saved attention mode/pooling and checkpoint temperature; Python 3.12 required; authored criterion order and numeric Score values retained through Choice transport |
| `jev27` | `autotrust/JEV-27B`, bundled converted `adapter_vllm` | Native bare-v1 single-pass readout through vLLM, all 2–256 labels, trained slot biases and saved Choice temperature; explicit server host/port required |
| `flymy` | FlyMy packaged pointer or letter-logit releases, bundled `model.py` | Manifest verification on a temporary real-file copy; native calibration; explicit context and letter capacity; pointer request byte cap raised from 64 KiB to 1 MiB while retaining structural validation and strict token limits |
| `lev` | `franckverrot/lev-350m`, franckverrot/lev | Native FP32 pointer model, pinned base revision, checkpoint calibration, strict encoding |
| `gliner2` | Fastino GLiNER2 checkpoints, fastino-ai/GLiNER2 | Native classification softmax with all labels returned; checks full schema-plus-text token count |
| `reranker` | Dedicated sequence-classification rerankers, Qwen3-Reranker, MXBAI v2 and ZeroRank 2 | Full rubric and one candidate per relevance pair; softmax of native relevance scores at temperature 1; this is an adapter distribution, not native decision calibration |
| `metask` | `wayfind/metask-jev-4b-policy-mix`, metask-ai/metask-jev `inference/` | Native enum prompt and released per-task temperatures; authored Noul definitions and numeric levels remain explicit |
| `spark` | `abhishek085/spark-s1-4b-v6`, abhishek085/open-spark-jev | Native menu prompt and per-task calibration; Noul uses explicit Choice options to retain authored definitions |
| `evalengine` | `evalengine/decision-4b`, checkpoint directory | Pinned base plus LoRA; published Tev-compatible JSON prompt and allowed-letter softmax |
| `deem` | `LibertAIDAI/deem-0.8-v1`, Libertai/deem | Native Torch backend and typed prompt; one original option ordering; default temperature 1 when the checkpoint has no calibration artifact |
| `hopper` | `HopitAI/hopper`, hopit-ai/hopper | Native calibrated letter probabilities, full authored Noul criteria, no shortlisting |
| `reflex` | `kshetrajna12/reflex-qwen3.5-4b-lora`, kshetrajna12/reflex | Pinned base and LoRA, native calibrated branches, one option ordering; probability sums normalized only within native six-decimal rounding error |
| `verdict-small` | `Manav2op/verdict-small`, Manavarya09/verdict | Native overlapping state windows and embedding similarity; option token overflow is rejected |
| `opendecision` | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0`, deepanwadhwa/OpenDecision | Native typed NLI scoring; full tokenization without truncation; Noul hypotheses retain instructions and definitions; authored hypothesis braces stay literal |
| `gliformer-jeff` | `knowledgator/gliformer-large-v1`, logan-markewich/jeff | Native isolated classification groups and temperature normalization, with complete-label validation and no output rounding; distinct from Firelex Jeff |
| `nimble-lora` | `jsaurabh/qwen3.5-9b-jev-data-mix-v2`, frozen bespokelabsai/nimble | Native schema candidate logits, pinned base plus LoRA, temperature 1 |
| `imajev` | `mohit67890/imajev-4b`, mohit67890/imajev | Native CUDA readout and released calibration, one option order; retain the unknown prompt branch and condition probabilities on declared benchmark options |
| `smalljev` | `isHeSatoshi/smalljev-semantic-v9`, isHeSatoshi/smalljev | Native shared option-span scorer; construct the complete prompt before checking capacity, disabling native state shortening |
| `plumb` | `crh225/plumb-4b`, crh225/plumb | Released single-read JevK5 protocol and checkpoint temperature; no confidence commitment heuristic |
| `rune` | Rune v3 BF16, invergent-ai/surogate | Decisions v1 prompt and filtered single-token codebook, Transformers CUDA logits, temperature 1, no thinking or order averaging; all-choice transport preserves criterion order and numeric Score values |
| `standardone` | `StandardThinking/StandardOne-8B`, bundled `server/` | Native wording and tokenizer boundary, no system prompt, released per-task temperatures, one option order |
| `jevone` | `juspay/jev-one`, bundled serving archive | Native 255-marker prompt and two-order reduction; released task temperatures, no output rounding; non-chat role records preserved as complete JSON |
| `needle` | `Cactus-Compute/needle3`, cactus-compute/needle | JAX CUDA teacher-forced likelihood of each complete tool-call candidate, normalized over declared candidates; this is an adapter distribution, not Needle's native confidence scalar. Parameter counts use the loaded Flax tree, excluding token and Engram embedding tables from static Active Params, even when shared with output heads; all loaded heads are included |

All these adapters accept `--context-limit`; omitting it retains the adapter's
recorded default. Inputs that exceed the effective limit fail rather than being
silently truncated. `jevlite`, `flymy`, `metask`, `spark`, `deem`, `hopper`,
`reflex`, `nimble-lora`, `smalljev`, `plumb`, and `standardone` also accept
`--max-candidates` for an explicit answer-code capacity extension. Extended
codebooks preserve the native prefix and require distinct single-token codes.
Smalljev's option-span scorer uses display labels instead of answer-token IDs;
its extension preserves the original alphabet and adds two-letter labels.
Extensions are recorded as evaluation conditions and do not imply training at
those lengths or candidate counts. Eval Engine uses the existing ordered Tev
24-way decision tree for larger menus, preserving every leaf and multiplying
conditional probabilities.

GLiNER2 explicitly requests its supported FlashDeBERTa backend and records the
actual encoder implementation. Its FP32 biased-attention kernel uses 32-by-32
tiles, one stage and four warps to fit the GPU shared-memory limit; this changes
kernel scheduling, not the input or checkpoint precision. Rerankers with a
published `LogitScore` configuration use their bundled query/document chat
template and token IDs. XLM-R rerankers also enforce the learned position-table
capacity, even when a larger runtime context was requested; overlong examples
fail explicitly instead of indexing beyond the embedding table.
Encoder rerankers batch up to eight candidates within an 8,192-token padded
microbatch budget. Decoder rerankers retain single-pair inference because padded
BF16 batching materially changed Qwen3 candidate logits in numerical checks.
Every candidate is length-checked before any candidate is evaluated.

Public source licenses and checkpoint licenses remain separate. Native runtimes
are imported from their recorded source checkout; they are not redistributed as
part of the S1MB package. The GLiNER2 project declares Transformers 4 compatibility;
use an isolated compatible runtime when necessary rather than modifying the
shared evaluation environment during a running job.

Use the `decision-encoders` extra for sentence-transformers, GLiFormer and
FlashDeBERTa. The `standardone` extra supplies its native `mistral-common`
tokenizer. Needle requires a separate CUDA JAX environment with `jax[cuda13]`,
Flax, Optax, SentencePiece and Safetensors, plus S1MB. It refuses CPU inference.
Needle's complete-candidate likelihood scoring is explicitly different from the
label-only native tool-call API: neither a one-hot label nor its single confidence
scalar is presented as a probability distribution. Its scores and latency must
be interpreted under that recorded adapter contract.

## Cloudflare Clef and Clef-Flash

Use `--adapter clef` for `Cloudflare/clef-flash` and `Cloudflare/clef`. Install
`uv sync --locked --extra open-models --extra meta-encoder`. The adapter loads
code, backbone, processor and joint head from the same resolved model SHA;
no separate source checkout is required. Text-only inference uses BF16 and SDPA
on an explicit CUDA device, with one case per forward pass and all its questions.
Supported GPU delta-rule kernels accelerate the Qwen recurrent layers; convolution
and rotary operations use native Torch GPU implementations.
The native 16,384-token default can be changed with `--context-limit`. Inputs
are fully encoded and overflow is rejected before inference; native state
truncation is disabled. Choice keys and question IDs are anonymous; zero-padded
choice keys preserve authored order under the native lexical sort. Structured
instructions and descriptions, authored Noul definitions and numeric Score values
are preserved. Score descriptions include their declared numeric values because
the native score interface otherwise represents only zero-based ordinal indices.

```sh
CUDA_VISIBLE_DEVICES=1 uv run --no-sync s1mb run \
  --adapter clef --model Cloudflare/clef-flash --revision MODEL_SHA \
  --device cuda --context-limit 32768 \
  --category smoke-v1 --limit 2 --run-id clef-flash-smoke-001
uv run --no-sync s1mb validate data/results/clef-flash-smoke-001
```

After validating smoke results, use `--category english-v1` without `--limit`
and a fresh run ID. Keep `--context-limit 32768` for the current English category:
its joint UD EWT schemas include cases longer than 16,384 tokens. This increases
the encoder cap; it does not change weights or truncate inputs. Repeat for
`Cloudflare/clef` with its own pinned revision.

## Julia and Dinah

`julia` loads `SupersonicLabs/Julia-1`'s pinned `TransformerEngine` directly,
without its optional router or CPU backend. It uses native marker serialization,
8192 total tokens, a 512-token question/option head, at most 20 candidates and at
most 48 tokens per option. Native `sequence(strict=True)` checks the full request
before inference and passes that exact encoding to the native collator. Requests
outside these bounds fail rather than being shortened or split into tournaments.
The release declares `transformers>=5.0,<5.1`; prepare a separate environment
when evaluating it instead of changing an environment used by another run.

`dinah` loads `Lukitaduarte/dinah-0`'s pinned Torch API on explicit CUDA with SDPA,
FP32 weights and BF16 autocast. Each question is a separate bounded call. Its
native encoder rejects inputs above 8192 tokens. The model's API supports Score,
although the supplied Decision Index wrapper only admits Choice and Noul.
S1MB uses the native Score probability distribution, preserving declared rubric
order and including each numeric level alongside its authored description.
This is recorded as a rendering condition; the native ordinal expected value is
not used as a replacement for S1MB's numeric levels.

Both adapters retain structured state/instructions/criteria, anonymize Choice
transport labels, preserve authored Noul definitions, and record pinned source
and checkpoint provenance. Neither loads targets or case IDs into model text.
Use `uv sync --extra open-models` for the common Torch runtime dependencies,
subject to Julia's separate Transformers requirement above.

`dm-jepa` loads `DangerLabs/DM-JEPA`'s pinned native latent verifier with FP32
weights, BF16 autocast and SDPA. The ModernBERT configuration and tokenizer are
independently pinned to `8949b909ec900327062f0ebf497f51aef5e6f0c8`; the adapter
constructs that backbone locally and loads the decision checkpoint strictly.
Native state and option formatting retains structured instructions and criteria,
anonymizes Choice labels and renders authored numeric Score levels. Full
tokenization precedes CUDA transfer: states exceeding the release's 16384-token
budget or criteria exceeding 512 tokens fail without truncation. The backbone's
declared position budget is recorded separately from the release budget.

```sh
CUDA_VISIBLE_DEVICES=1 uv run s1mb run \
  --adapter julia --model SupersonicLabs/Julia-1 \
  --revision a85b127321d580d65176c89ced8273f305745d85 \
  --device cuda:0 --category smoke-v1 --limit 2 \
  --run-id julia-smoke-001

CUDA_VISIBLE_DEVICES=1 uv run s1mb run \
  --adapter dinah --model Lukitaduarte/dinah-0 \
  --revision 07c6884439df7c3d2cd01eea16924ed07b866dea \
  --device cuda:0 --category smoke-v1 --limit 2 \
  --run-id dinah-smoke-001
```

Choose fresh IDs if these directories already exist. Inspect GPU 1's free memory
and wait for its current workload to finish before running either command.
Validate each saved smoke run before moving on to `english-v1`.

## KnowLine (PelaAI)

Use `--adapter knowline` for `PelaAI/KnowLine-4B-Gen2` and later KnowLine releases. The model is served by its own
`/v1/systemone` server from the model repo (`serve_knowline.sh`: SGLang with FP8 at load, then `knowline_server.py`,
chat style, temperature 1), which implements the TypeSafe v1 typed-question contract. The adapter reuses the TypeSafe
request mapping (structured instructions, ascending Score levels) with a local base URL from `KNOWLINE_BASE_URL`, no
credentials and unrounded probabilities. The server takes up to 64 questions per request, as Jev does; cases with more
questions are sent in chunks of 64 with the same state. Each question is scored with its own prefill, so chunking does
not change any answer. `--case-batch-size` (default 16) sets how many cases are sent concurrently.

```sh
# in a clone of the model repo, on one GPU
bash serve_knowline.sh PelaAI/KnowLine-4B-Gen2 0 8080
# from evaluator/
KNOWLINE_BASE_URL=http://127.0.0.1:8080 uv run --no-sync s1mb run --adapter knowline \
  --model PelaAI/KnowLine-4B-Gen2 --revision MODEL_SHA \
  --category smoke-v1 --limit 2 --run-id knowline-smoke-001
uv run --no-sync s1mb validate data/results/knowline-smoke-001
```

Use a `knowline_server.py` from 2026-10-08 or later (model repo commit `d62c958` for Gen2). Earlier copies close the
connection on chats whose roles the chat template rejects (for example `customer` / `agent`), which fails one case of
`s1mb-generalization-diverse-score-score-test-v1`.


## JevK5-Lite

`jevk5-lite` is separate from the autoregressive `jevk5` adapter. Pass a
checkout of `allebee/jevk5` v0.3.1 (commit
`c6d1b01b194d2f36c8be129159c6bd270f3510b4`) with `--source`, a pinned model
revision, and an explicit CUDA device. The adapter loads the encoder and scorer
in FP32 and uses the native single-label calibrated softmax for all three tasks.
It preserves declared option order, structured authored definitions and numeric
Score levels; anonymous positional prefixes distinguish duplicate definitions.
No dataset identifiers, targets or provenance enter model text.

The native runtime truncates the body to fit its 512-token budget. S1MB compares
the native encoded body with complete tokenization and checks the total sequence
length before inference, rejecting any overflow. `--context-limit` may lower
this budget. Overflow remains a failed decision in saved results. The native
DeBERTa attention backend is recorded; there is no CPU fallback.

```sh
CUDA_VISIBLE_DEVICES=1 uv run s1mb run \
  --adapter jevk5-lite --model alibiserikbay/JevK5-Lite \
  --revision 315ee211f828a899161477c534cea23e84ba3568 \
  --source /path/to/pinned/jevk5 --device cuda:0 \
  --category smoke-v1 --limit 2 --run-id jevk5-lite-smoke-001
uv run s1mb validate data/results/jevk5-lite-smoke-001
```

Use fresh IDs and validate the smoke results before a full category run.


## Lavoir

`lavoir` requires a pinned `moganai/lavoir` source checkout, a pinned checkpoint
revision and explicit CUDA. It loads the complete decision and VOI checkpoint
strictly, then evaluates only slot-free decisions. The native default general-data
calibration is retained; no source-specific calibration or clarification policy
is selected. FP32 weights, BF16 autocast and SDPA match the native CUDA API.

The adapter keeps the slot-free sequence layout and option segment embeddings,
while tokenizing complete fields and rejecting overflow instead of applying the
native truncation. The total budget is 1024 tokens (lowerable with
`--context-limit`) and the native slot-free head budget is 256. Choice labels are
anonymous, Noul keeps both authored poles and Score descriptions include their
numeric levels. Each question is one bounded inference call.

```sh
CUDA_VISIBLE_DEVICES=1 uv run s1mb run \
  --adapter lavoir --model moganai/lavoir \
  --revision 4c5eaeb99b2368f1416b9893c7aacc58235aa09f \
  --source /path/to/pinned/lavoir --device cuda:0 \
  --category smoke-v1 --limit 2 --run-id lavoir-smoke-001
uv run s1mb validate data/results/lavoir-smoke-001
```

Install `laya==0.3.20` alongside the common open-model dependencies. Validate
saved smoke files before running `english-v1`, with a fresh run ID.


## LFM RLCD

`lfm-rlcd` evaluates the release's bundled unchanged LFM2 weights with its pinned
native prompt, JSON-value token boundary and full candidate log-likelihood sum,
including the newline terminator. Supply the snapshot's source directory using
`--source`. The adapter forks the native hybrid attention/convolution cache in
bounded groups of at most eight candidates and 32,768 padded cache-plus-branch
tokens. Complete prefix-plus-value paths are checked before inference; the
32,768-token default may be changed explicitly up to checkpoint capacity.

The native API exposes log-likelihoods, not calibrated probabilities. S1MB uses
a temperature-one softmax over full value likelihoods, without length
normalization; saved metadata explicitly labels the distribution uncalibrated.
Choice transport labels are anonymous, Noul retains authored definitions, and
Score enum values retain numeric levels and authored descriptions. Instructions
and structured state retain the dataset defaults.

FP32 and SDPA preserve branch-batch stability: a GPU parity check against the
native all-candidate implementation found a maximum probability difference of
1.91e-6 across synthetic Choice, Noul and Score requests, including a ten-option
Choice spanning two bounded batches. BF16 produced a material batch-size
sensitivity and is not used for this adapter.

Install common open-model dependencies and `--extra lfm-rlcd`. Use a fresh
three-task smoke, validate its files, then evaluate the full category with a new
run ID. Bundled base weights and source/checkpoint digests are recorded; there is
no independent unpinned base-model load or CPU fallback.

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

### this-that 1.2 native bridge

The `thisthat` adapter uses the pinned FLock `TypedDecider` PyTorch backend,
BF16 with SDPA, the native answer-slot restricted-label readout, state-first
layout, and the native default temperature of 1.0. Supply the source checkout
with `--source`; checkpoint and source Python digests are recorded in metadata.
Each question is evaluated separately, with authored criteria in declared order;
structured descriptions/instructions are serialized as JSON and Score options
include their numeric values. The native maximum is 255 distinct option texts.
Duplicate option texts are rejected by the native constructor, without adding
invented distinctions. The bridge overrides the native 1,536-token state slice
with the exact complete state token count and rejects full padded inputs over
32,768 tokens before GPU inference. This is an explicit full-state configuration.
CPU mapping/boundary tests pass; GPU parity, smoke and full-run validation are
required before claiming measured support. No HTTP service is launched.

### EXAONE-JEV native bridge

`exaone-jev` loads the pinned `serve/systemone_server.py` in a private module and
retains its question validation, tev1 rendering, calibration by task/candidate
count, original/reversed permutation averaging, balanced candidate chunks and
final-pool mass redistribution. All original candidates remain in the returned
distribution. Only the label-logprob transport uses Transformers BF16/SDPA, as
supported by the checkpoint card, instead of vLLM. The full input limit is
explicitly 32,768 tokens, including the native one-token output reservation;
overflow is rejected without truncation. Structured authored instructions and
criteria are preserved; Score descriptions include numeric values. Install
`uv sync --locked --extra open-models --extra exaone-jev` and supply the pinned
source checkout with `--source`. Importing native helpers does not start an HTTP
service. CPU tests and static checks are prerequisites; GPU parity, validated
smoke and full results must pass before measured support is claimed.

### Jev-Style 2B v3 block-causal bridge

Use `jev-style-2b` for the pinned 2B release. It shares typed question mapping
with the 0.8B bridge but has a distinct native runtime: 2,048-token block-causal
attention in full-attention layers, recurrent state in Gated-DeltaNet layers,
and a complete catalogue/rubric renderer for long options. The adapter preserves
`jev_style_block_sdpa`; selecting ordinary causal SDPA would change the model.
FP32 inference/readout, native global temperature (0.8278650620942867), manifest
verification, a 25,600-token complete-input limit, and no truncation are retained.
CUDA graphs are off. Native state-cache reuse/deep-copy behavior remains enabled;
close releases the cached state before freeing the model. Supply the pinned
checkpoint snapshot as `--source`. GPU parity must include a long catalogue with
more than one block, then validated smoke and full results before measured
support is claimed.

### Sieve native fork bridge

`sieve` uses the pinned native Sieve constructor, LoRA, scalar head, sorted
listed-question catalogue, isolated recurrent/attention prefix forks and released
calibration temperature. Specify `--base-revision` for the dependency checkpoint;
the exact base SHA is resolved once and recorded alongside the adapter revision.
The base is a dependency, not another evaluated leaderboard row. The bridge
uses supported native FP32 with SDPA. It rejects complete prefix-plus-branch
input overflow before any model forward, without calling native `_trim`.
Candidates are scored in bounded groups (at most eight branches and a 32,768
branch padded-token budget); each group receives an independent native prefix
cache and all logits enter one global softmax. No candidate is filtered.
Structured descriptions/instructions and authored Noul definitions remain
intact; Score options include numeric values and use native Choice scoring.
Native indistinguishable duplicate option texts remain unsupported. No
benchmark-specific schema fitting is performed. GPU parity must compare full
native scoring and option permutations, then smoke/full validation must pass
before measured support is claimed.

### LFM2.5 2.6B PCD native bridge

The 2.6B RLCD release bundles the original LFM weights and its distinct `pcd`
package. Use `lfm-pcd`, the `lfm-pcd` optional extra, and the pinned snapshot as
`--source`. The adapter uses native token mode: atomic option codes, the native
catalogue/prompt, one cached branch per field and restricted output-head scoring.
It preserves structured authored definitions and numeric Score values via the
same anonymous schema inputs as the 350M bridge. Probabilities are the native
uncalibrated restricted-code softmax, not full-sequence likelihoods.
Supported native Limits are explicitly configured: 32,768 complete input tokens,
32,768 serialized-value tokens, bounded input/schema character budgets, one field
branch per call and 32 positions per projection. Record these limits in metadata;
complete prefix plus field suffix is checked before prefill. Native conservative
GPU memory admission remains active. FP32/SDPA is used, with the bundled release
weights. GPU parity must compare the native cached output and uncached full-head
oracle, validate convolution/cache immutability, and include a 151-option choice.
CPU input audits do not prove GPU memory admission or measured support.

### Kodiak accuracy ensemble native bridge

`kodiak` loads all three pinned ensemble members through the native Kodiak hub
loader and applies their released calibration. Supply the source checkout and
install the `kodiak` optional extra. Bundled tokenizer files must be identical;
the bridge binds native tokenization to those local files without a separate
unpinned ModernBERT download. Native FP32 weights/BF16 CUDA autocast and packed
block-mask SDPA remain intact. Schema validation and native packing happen before
any GPU forward; any native truncation flag is rejected. Native request limits
(including 32 choice labels and 200 characters per label) are retained.
Choice and Noul use authored label text, anonymous transport IDs, and explicit
`allow_null=False`, returning native calibrated conditional probabilities.
Score uses the native calibrated ensemble's moment-matched Beta over its numeric
range, includes every authored numeric level and description in the question,
and supplies native endpoint anchors. The bridge discretizes this continuous
Beta into the declared numeric levels using midpoint boundaries. This is an
explicit score-distribution interpretation; it is not an ordinal accuracy or a
replacement classifier head. No labels are removed to fit the request limits.
GPU parity must cover the public native ensemble probabilities and Score mean/
variance, then validate three-task smoke and complete traversal results. CPU
schema/token audits do not establish measured support.

### Lev1 3.7B native bridge

`lev1` loads the pinned release's bundled native scorer and merged weights with
`--source <pinned-snapshot>`. BF16 packed block-mask SDPA, native cached long-input
scoring, and original/reversed option-order probability averaging are preserved.
The explicit default complete-input limit is 32,768 tokens; overflow is rejected
before inference. Authored Noul definitions and numeric Score values are passed
as anonymous native Choice criteria. Native temperature defaults to 1 when no
calibration is provided. GPU parity must exercise the cached long-option path as
well as all three task mappings before validated smoke and full traversal.

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
Use the native server default complete-input path, then explicitly set the
encoder budget (default 32,768) and retain `truncate=none`. The release loader
accepts only training cut policies, so do not pass `none` through its environment
overrides. Restore the caller environment after construction. Use explicit Choice criteria
for authored Noul definitions and numeric Score levels. Preserve the trained
option pooling and every candidate, reject any truncation report, and require
native GPU parity plus validated smoke before full traversal.

Gevva uses `--adapter gevva --source <pinned-gevva-checkout>` for the self-contained
E2B and E4B releases. Preserve the native three-class NLI calibration and native
bucketed margin softmax through `GevvaCrossEncoder.decide`. Encode all authored
Noul criteria and numeric Score levels as explicit Choice descriptions. Before
inference, compare the native bounded pair tokenization with complete pair
encoding and reject any difference; never silently cut state or hypotheses.
Candidate batches are bounded at four. Require native GPU probability parity,
validated three-task smoke, and saved-result validation for each pinned release.

Winnow E4B selects the Q8 artifact from the pinned runtime release-assets manifest
and verifies its exact size and SHA256. Pass the positive finite
direct-text Q8 calibration from the verified E4B release card through the native
`winnow.temperature` request extension;
record it in metadata. Explicit Choice transport preserves authored Noul
definitions and numeric Score levels without replacing them with ordinal labels.

Jevstral uses `--adapter jevstral --source <pinned-jevstral-checkout>` and the
calibrated `stage4/final` release. Its native code pins the Ministral 3 base SHA;
require the checkpoint configuration to match it. Use the native bf16 merged
LoRA decoder with trained delimiter rows, fp32 pointer head and saved temperature.
Keep native serving limits (65,536 state tokens and 73,728 row tokens), or use a
stricter explicit context limit. Native encoding rejects overflow. One question
per call avoids duplicating state across batches and uses the native uncached
path. Explicit Choice criteria preserve authored Noul definitions and numeric
Score values. A separate Transformers 5.18 environment avoids changing the RSI
and Gevva runtimes. Require native GPU parity and validated smoke before full runs.

Nimble v2 uses the released `ParallelScorer` from the pinned model snapshot,
including its T=2.179078721266035 calibration and native serving codebook.
Nimble v3 uses `--adapter nimble-v3 --source <pinned-evaluation-snapshot>/code`;
its released `nimble.evaluation.decision_index_engine.NimbleEngine` verifies the
prompt hashes and 255-token codebook against the adapter contract. Use the
contract's exact Qwen base revision, native unscaled softmax, SDPA and one question
per call. Explicit Choice transport keeps authored Noul descriptions and numeric
Score levels. Complete prompts are rejected beyond the explicit budget (default
32,768). Require native GPU parity, full input audit and validated smoke before
full traversal. The v3 weights retain their separate CC BY-NC 4.0 license.

JevK5 9B uses the pinned native raw probability API with its released temperature
and separate knockout temperature. Explicit Choice criteria preserve structured
descriptions, authored Noul definitions and numeric Score values. Native knockout
scores every candidate in groups above 16 choices. Check every actual branch's
complete chat-template tokenization against 32,768; reject overflow before model
forward. Use SDPA, disabled CUDA graphs, and native probability parity including a
151-candidate fixture before validated smoke and full traversal.


### AutoTrust JEV 9B

Use `--adapter autotrust-jev` with an immutable checkpoint revision and `--source`
pointing to that release snapshot. This follows the release card's Transformers
inference example: load the bundled BF16 text backbone, merge its released LoRA,
and apply the separate FP32 24-slot head to the last final-norm hidden state.
Use SDPA and the released Choice calibration temperature. The bare-v1 Choice
transport preserves structured descriptions, authored Noul definitions, criterion
order and numeric Score levels. It supports 2–16 candidates; wider questions fail
explicitly without dropping candidates. Reject complete-input overflow before
inference (default 32,768 tokens, bounded by checkpoint position capacity).
The separate slot head is included in parameter metadata. Require independent
model-card probability parity across all three tasks before smoke and full runs.


### Sieve 9B

Use `--adapter sieve-9b` and `--source` pointing to the upstream Sieve package.
This release differs from the Sieve 2B scalar-head checkpoint. Load its BF16
merged LoRA text backbone and FP32 pointer head with the base revision recorded
in the release's provenance.json. The native Decision Index single-question
path scores one complete causal row, with the released head temperature and
SDPA, without CUDA graphs or cached state reuse. Preserve all criteria using
structured Choice transport, including authored Noul definitions and numeric
Score levels. Native admission limits are 65,536 state tokens, 32,768 branch
tokens and 255 candidates. An optional context limit also bounds the complete
row without truncation. Verify probability parity against the upstream Decision
Index engine before validated smoke and full evaluation.


### AJev Gemma 4 12B LoRA5

Use `--adapter ajev`, a pinned checkpoint revision, `--source` pointing to the
upstream ajev-infer package, and an explicit `--base-revision`. Load the BF16
base and unmerged released LoRA in memory; use SDPA and native bare/space label
logsumexp scoring with the checkpoint's per-type temperatures. Native typed
hints are retained. Preserve declared criterion order, authored Noul definitions,
structured values and numeric Score levels in option descriptions. Disable state
truncation and reject complete native prompts above the context limit (default
32,768, bounded by checkpoint capacity). Score one case per call without shared
prefix caching. Require native probability parity and validated smoke before
full runs. Do not save and reload a merged Gemma 4 checkpoint.


### Xor 26B-A4B

Use `--adapter xor`, an immutable model revision and `--source` pointing to the
verified release's serving bundle. `--server-host` and `--server-port` select the
released SGLang backend, which must load the verified snapshot at `/models/xor`.
Use the image digest specified by the release (including its logprob fix), one
explicit GPU, a safe bind address and a fresh cache before each evaluation run.
The adapter calls the released compatibility server in process; it does not
start or stop the backend. It preserves structured definitions, authored Noul
criteria, numeric Score levels and criterion order using native Choice transport
and the released Choice temperature of 1.95. Native admission rejects overflow
and more than 255 candidates. Validate native probabilities across all three
tasks, then smoke and complete runs before presenting results. Parameter metadata
uses immutable checkpoint headers with the shared static embedding exclusion.


### Blink v0.3 26B-A4B NVFP4

Use `--adapter blink`, a pinned NVFP4 release, `--source` pointing to the author's
Blink source, and `--server-host` / `--server-port` for its vLLM backend. Load the
verified snapshot at `/models/blink`, served as `blink`, using `modelopt_fp4`, FP8
KV cache, disabled thinking and the release's 32K context. The adapter renders
surogate decisions v1, preserving authored Noul criteria, structured descriptions,
numeric Score levels and declared criterion order. It checks each complete prompt
including the answer token and validates single-token label continuations before
requesting every candidate's `logprob_token_ids`. Missing probabilities are errors;
the release's top-20 demonstration is insufficient for wide distributions.
Apply the released temperature, 1.3. Parameter metadata uses a native empty-model
logical census; verify packed NVFP4 and per-expert shapes against that census
before evaluation. Native source rendering/probability parity, three-task smoke,
full traversal and saved-result/data validation are required. Backend lifecycle
belongs to the runner; use only the selected physical GPU and safe bind address.


### AutoTrust GEV 26B Decide

Use `--adapter gev`, a pinned release and `--source` pointing to its snapshot.
Use the released Transformers System 1 path: BF16 base plus in-memory merged
LoRA, BOS-prefixed bare-v1 template and FP32 24-slot head with softcap 30 and the
default calibration.json. Explicit Choice transport preserves authored Noul
definitions, structured descriptions, numeric Score levels and criterion order.
The released server's `_groups` and `s1_dist` orchestration functions are loaded
verbatim without its vLLM server imports. Native tournament scoring reads every
candidate in groups of at most 16, then refines 16 finalists while retaining
probability mass for every original option (up to 256). Each HF branch is scored
sequentially to bound GPU memory. Reject complete branch overflow before inference
(default 32K; an explicit override is bounded by checkpoint capacity). Adaptive
thinking is disabled. Verify model-card head parity for all tasks and native
151-candidate tournament parity before validated smoke and full traversal.

Torchcast Decision 27B: the `torchcast` bridge loads the pinned bundled native `torchcast_decision` package supplied through `--source`. It uses the native causal decoder, FP32 letter readout, and wide-choice tournament, with task-specific calibration. Evaluation uses a 32,768-token input limit, disabled CUDA graphs and images, and supported Transformers GPU kernels. Each actual tournament row retains native overflow rejection. Structured Score values and authored criterion order are carried through anonymous Choice criteria.

Solomon: the `solomon` bridge loads the pinned author's `ServiceEngine` and validates its serving binding before prediction. The bundled release source is supplied through `--source`; the external Qwen base revision is read from the release manifest. Keep question-side FP32 LoRA unmerged, prefill the document with the adapter disabled, then branch through native trained heads. Explicit Choice criteria preserve authored Noul definitions; numeric Score criteria use the ordered S head. Native Choice capacity is 2–8 candidates and wider questions are rejected. The bridge rejects full prompts beyond 32,768 tokens before prefill. Native GPU parity and smoke checks remain required before full evaluation.

Eikos FP8: `eikos-fp8` uses the released semif JSON renderer and a verified vLLM >=0.30 FP8 dynamic backend. Supply pinned native code through `--source` and the backend address through `--server-host` / `--server-port`. All selected candidate log probabilities must be present; missing tails are rejected. Structured and numeric criteria preserve authored order through explicit Choice transport. The runtime uses a 32,768-token limit including one answer token, identity calibration and hybrid prefix-cache alignment. Native GPU parity and smoke checks are required before full evaluation.

JADE: `jade` uses the release's exact prompt, decision vocabulary, temperature and exported rank-256 vLLM adapter. Supply a materialized, checksum-valid release through `--source` and the backend address through `--server-host` / `--server-port`. Read requested log probabilities in native 128-token chunks over the same complete option set, then calibrate once. Keep its hard limit of 8,192 tokens including one answer token; larger inputs raise errors. Explicit Choice transport preserves authored Noul definitions and numeric Score criteria. GPU parity and smoke validation remain required before full runs.

## Caller-managed llama.cpp

The `llama-cpp` adapter connects to a running llama-server's native
`/v1/systemone` endpoint. Start and stop the server yourself; the adapter does not
build llama.cpp, download weights, start a process, or choose a GPU. It requires
a native decision GGUF supported by that endpoint. Ordinary chat GGUFs are not
supported by this adapter. No optional Python runtime extra is needed.

Start your independently obtained model with a compatible llama.cpp build:

```sh
# On the shared workspace, inspect GPU 1 availability first.
CUDA_VISIBLE_DEVICES=1 llama-server -m /path/to/decision.gguf \
  --alias decision-model --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 16384 -b 16384 -ub 16384 --parallel 1 --no-context-shift
```

Choose context and batch sizes for the model and available GPU memory; these
numbers are examples, not universal settings. When Tailscale is available, bind
to that machine's Tailscale IPv4 address instead of localhost. Pin the model
revision, GGUF hash, llama.cpp revision, precision, and effective server settings.
The adapter does not verify an externally started server's weights against a Hub
revision; `--revision` records the caller-declared checkpoint revision, while
`served_model` verifies the response alias and detects model-name changes.

From `evaluator/`, evaluate with constructor kwargs supplied as JSON:

```sh
uv run s1mb run --adapter llama-cpp \
  --model organization/decision-model --revision CHECKPOINT_COMMIT \
  --adapter-kwargs '{"base_url":"http://127.0.0.1:8080","served_model":"decision-model","timeout":600,"case_batch_size":1,"max_candidates":52,"runtime_settings":{"llama_cpp_revision":"SERVER_COMMIT","weights_sha256":"GGUF_SHA256","dtype":"Q8_0","context_size":16384}}' \
  --category smoke-v1 --limit 2 --run-id llama-cpp-smoke-001
uv run s1mb validate data/results/llama-cpp-smoke-001
```

For Python use, the same options are ordinary keyword arguments:

```python
from s1mb.adapters.llama_cpp import LlamaCppAdapter

kwargs = {
    "base_url": "http://127.0.0.1:8080",
    "served_model": "decision-model",
    "timeout": 600,
    "case_batch_size": 1,
    "max_candidates": 52,
    "runtime_settings": {"llama_cpp_revision": "SERVER_COMMIT", "dtype": "Q8_0"},
}
adapter = LlamaCppAdapter("organization/decision-model", "CHECKPOINT_COMMIT", **kwargs)
try:
    predictions = adapter.predict(inference_case)  # an s1mb.data.InferenceCase
finally:
    adapter.close()  # closes the HTTP client; leaves llama-server running
```

Supported kwargs are `base_url`, `served_model`, `timeout`, `case_batch_size`
(1..32), `max_questions` (1..64), `max_candidates`, `max_request_bytes` (default
16 MiB), `retries` (0..5), and `runtime_settings` (recorded metadata only).
`base_url` may contain a proxy path prefix. If the server requires a bearer token,
set `S1MB_LLAMA_API_KEY` in a local ignored environment file. Do not put credentials
in JSON kwargs, URLs, or runtime metadata. Changing concurrency may affect backend
arithmetic; record it and smoke-test it before a full run.

The adapter sends all questions of a case together, preserves joint-model
semantics, and rejects oversized cases rather than splitting their questions.
Question and Choice transport keys are anonymous. Structured state, instructions,
and criteria use the native typed contract. Score levels are sent in ascending
numeric-value order and response probabilities map back to the original option
IDs; the original numeric values remain evaluator-side. The native API supports
2..10 Score levels. Probabilities must validate without client renormalization.
Request counts and token usage are recorded, and bounded retries apply only to
transient failures. Batch failures remain individual failures.

The native server renders/tokenizes the model-specific prompt and must reject
context/batch overflow **before inference**. The client cannot pre-count that
exact prompt through this API, does not truncate text, and rejects request-byte
and configured candidate/question-capacity overflow locally. Server overflow
errors are propagated without retry or fabricated probabilities. A generic
`/tokenize` call on serialized request JSON would not count the native prompt.
Do not use native runtimes that silently truncate, including the current Laya
and LFM2-D1-Omni native renderers, with this adapter for S1MB until a separate
no-truncation preflight is available. `--no-context-shift` alone does not disable
model-specific renderer truncation. Backend temperature comes from the model's
metadata; chat-generation parameters such as `temperature` and `top_p` are not
constructor kwargs and are not passed to `/v1/systemone`.
