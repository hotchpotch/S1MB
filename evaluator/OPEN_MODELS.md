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
| Jebadiah (`jebadiah`) | Native AINode renderer and saved per-task temperatures |
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

## The two Open-Jev implementations

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
