---
name: s1mb-evaluate-and-submit
description: Explicitly invoked workflow to evaluate self-developed, public-weight, or API models with S1MB and submit validated HF result PRs, one per distinct trained model. Maintainer merge and deployment are outside this workflow.
---

# S1MB Evaluate and Submit

Use only when explicitly requested. Read the project's `AGENTS.md` and
[evaluation guide](../../../docs/evaluation.md). Follow the
[submission guide](../../../docs/contributing_results.md) for packaging and PRs,
and [training disclosures](../../../docs/training_exposure.md) for known flags.
Resolve links relative to this skill directory; use current project schemas.

## Identify models and reuse adapters

Confirm each model's identity, exact checkpoint/source revision or resolved API
version, evaluation language, available resources and requested outcome. Check
existing leaderboard rows and open submissions. Prefer original public weights
when both weights and API are available; disclose exceptions and unknown identity.

Inspect existing adapters and [runtime notes](../../../evaluator/OPEN_MODELS.md).
Consider model-specific adapters, native llama.cpp `/v1/systemone`, OpenRouter and
other supported interfaces before implementing an adapter. Verify full input,
authored instructions, candidate order, Noul meanings, numeric Score levels and
probability output. Chat-completion support alone does not establish compatibility.
If unsupported, follow [adapter development](../../../docs/adapters.md), add
focused synthetic tests, and prepare an authorized source PR separately from
results. Link the two PRs. Inspect third-party code before executing it.

## One trained model per result PR

Different underlying trained models require separate HF Dataset PRs, even when
evaluated together. Genuinely different trained releases or fine-tunes need
separate PRs; a shared architecture or pretrained base does not establish identity.

Quantization, precision, runtime, API route, mirror and packaging variants of the
same trained model may share a PR, but still use one canonical model folder and
leaderboard row. Preserve run provenance, explain additions/replacements, and
never assemble a best-of-configurations row. Stage each trained model separately
with its own PR body. Comparison reference models must stay outside uploads.

## Freeze conditions and evaluate

Respect the requested scope and existing authorization. An explanation or
preparation request does not authorize costly inference or public PR submission.
Do not repeat permission questions for actions already authorized. Record scope,
model identities, frozen settings, run IDs, validation, paths and PR links under
ignored `tmp/`; keep credentials and private inputs out of reports/source commits.

Record model/source SHAs, Dataset selection, precision, batching, input policy and
calibration before evaluating. Never select checkpoints or tune settings using
S1MB test labels/scores; S1MB provides no validation split. Diagnose technical
failures without choosing the highest-scoring run.

Run Python commands from `evaluator/`. Check GPU availability and project device
rules, run smoke checks, then the requested full scope with fresh run IDs. Do not
silently fall back to CPU or truncate inputs outside an explicitly documented
adapter policy. Preserve original measurements, failures, incomplete results,
exact Dataset revisions and reasons for technical reruns.

## Validate and submit

Validate against each result's recorded Dataset revision with the current trusted
evaluator. Inspect coverage/failures before reporting scores; incomplete results
cannot claim complete task averages. Prepare metadata and supported training
exposure declarations with evidence, noting unknown details. Known flags do not
adjust scores or independently prove training histories. Missing means unknown,
not unseen; evaluation-example use is a distinct disclosure.

Check TP/AP and counting method: static AP excludes embedding weights, including
tied output weights, and is not routed per-token MoE parameters. Leave unsupported
counts unknown rather than estimating them from model names.

Export each model to its own staging root and run `validate-results`. Use existing
submission tooling to prepare a comparison/PR body with identity, effective
settings, provenance, coverage, reruns and training disclosures. Keep locks,
reports, reference models, source code, weights and private data outside uploads.

When submission is authorized, create one Dataset PR per trained model, uploading
only metadata and compressed measurements. Use the parent-SHA guard from the
submission guide, inspect returned file inventories and record exact PR heads.
For corrections, update the existing PR ref. Inspect concurrent changes before
retrying rejected writes instead of blindly repeating a mutation.

Report coverage/validation, source/result PR links and remaining issues. Hand off
review, merge, summary publication and Space deployment to maintainers; evaluation
or submission approval does not authorize those actions.
