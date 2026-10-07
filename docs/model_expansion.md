# Open model evaluation preparation

Compare the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
with the [published S1MB summary](https://huggingface.co/datasets/hotchpotch/s1mb-result/blob/main/viewer-summary.json).
Match exact checkpoint IDs and configurations, not display names alone. A missing
published row means a publication candidate; it does not establish that no local
smoke or unpublished evaluation exists. Check local results before scheduling.
Downloaded inventories, model cards, preflight reports and measurements belong in
ignored local artifacts, not this document or the source repository.

## Preparation order

1. Resolve each public HF model to an exact SHA, inspect release files and review
   its code/model licenses separately. Preserve the checkpoint ID, subfolder and
   quantization configuration; do not substitute a newer model for the listed one.
2. Inspect the effective HF cache filesystem and download the small candidates
   there first. Keep a disk reserve and log failed downloads. Downloading weights
   does not mean a model is runnable or supports every S1MB task.
3. Audit native rendering, calibration and input budgets. Reuse existing adapters
   only after establishing interface compatibility. New versions in an existing
   family can require a new adapter.
4. Implement lossless adapters and CPU boundary tests. Preflight the active
   manifest's inputs using the checkpoint tokenizer without loading GPU weights.
   Keep rejected inputs visible. Do not shorten inputs or weaken candidate bounds
   to obtain a complete leaderboard row.
5. Once GPU 1 is available, inspect free memory, explicitly expose only physical
   GPU 1, run `smoke-v1 --limit 2`, inspect metadata and validate the saved results.
   Then run a fresh full evaluation and validate against its recorded dataset SHA.
6. Export validated results for a results Dataset PR. Keep partial coverage
   incomplete; publication, summary updates and viewer deployment are separate.

## Initial small-model queue

Sizes below are approximate Decision Index metadata, not S1MB parameter counts.
Use measured `non_lookup_parameters_v1` metadata after loading. In particular,
JevK5-Lite's card reports 437M rather than the index's 300M, and GLiNER2.5-Decide's
card describes 340M rather than the index's 486M. Resolve these discrepancies
before treating this as an exact size ordering.

| Approximate size | Checkpoint | Preparation |
| --- | --- | --- |
| 141M | `SupersonicLabs/Julia-1` | Native adapter; strict 20-option / head-token bounds may prevent full coverage |
| 149M | `DangerLabs/DM-JEPA` | Audit separate state/criterion encoding and pin backbone configuration/tokenizer dependencies |
| 150M | `Lukitaduarte/dinah-0` | Native Torch adapter; native Score API differs from Decision Index wrapper |
| 300M in index | `alibiserikbay/JevK5-Lite` | Audit Lite runtime; existing autoregressive JevK5 adapter is insufficient |
| 354M | `notnotsamuel/LFM2.5-350M-RLCD` | Audit branch-scoring runtime and unchanged backbone provenance |
| 396M | `moganai/lavoir` | Audit Laya-derived decision interface and exclude clarification/VOI outputs |
| 486M in index | `fastino/GLiNER2.5-Decide` | Check existing GLiNER2 adapter against current native classification/calibration |
| 600M | `vllm-sr/Decision-2.0-Kai-0.6B` | Audit native constrained decision readout |
| About 870M | OneJev, Decision Eos, Jev-Style v3, Sifr v3, jiwo, JPT, Intern-Decision | Audit each released runtime before assigning adapters |

Prioritize auditing JPT-0.8B after the small encoder smoke checks: its model card
includes a 32768-context serving example. Verify the local native readout,
checkpoint calibration and lossless token budget before claiming S1MB coverage.

After this group, prepare the 1.2B–4B checkpoints before the larger dense models.
For MoE models, report both total and active parameters and assess VRAM using the
full stored model. Original Decision Index scores are not S1MB measurements.

See [external adapter instructions](../evaluator/OPEN_MODELS.md#julia-and-dinah)
for the new smoke commands and [evaluation](evaluation.md) for dataset locking,
validation and full runs.
