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
remain active under `non_lookup_parameters_v1`. The verified Q8 release
(`b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea`)
contains 11,907,350,576 total and active parameters across 667 tensors.
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
| `autojev` | `denis-pplx/AutoJev-27B`, bundled `source/src/autojev` | Native decision head and checkpoint temperature; Python 3.12 required |
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
| `rune` | Rune v3 BF16, invergent-ai/surogate | Decisions v1 prompt and codebook, Transformers CUDA logits, temperature 1, no thinking or order averaging |
| `standardone` | `StandardThinking/StandardOne-8B`, bundled `server/` | Native wording and tokenizer boundary, no system prompt, released per-task temperatures, one option order |
| `jevone` | `juspay/jev-one`, bundled serving archive | Native 255-marker prompt and two-order reduction; released task temperatures, no output rounding; non-chat role records preserved as complete JSON |
| `needle` | `Cactus-Compute/needle3`, cactus-compute/needle | JAX CUDA teacher-forced likelihood of each complete tool-call candidate, normalized over declared candidates; this is an adapter distribution, not Needle's native confidence scalar. Parameter counts use the loaded Flax tree, retaining tied token/output embeddings and excluding lookup-only Engram tables from Active Params; all loaded heads are included |

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
