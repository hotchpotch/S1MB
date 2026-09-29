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

Support for an adapter does not imply that a checkpoint is publicly available or
that every benchmark fits its context limit. Failed or partial runs remain visibly
incomplete. Do not interpret arbitrary upstream truncation or unavailable weights
as successful evaluation. Model and source licenses must be reviewed separately.

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
