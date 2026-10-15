# Diagnosing evaluation failures and input limits

Use this guide when a smoke check or full run leaves missing or partial results.
It describes the adapters currently implemented in this repository, not a promise
about a provider's current service limits. Check the evaluated checkpoint, pinned
native runtime and saved model settings before applying an example to another release.
The [runtime reference](../evaluator/OPEN_MODELS.md) owns installation and
model-specific configuration details.

**Any missing benchmark, unfinished task or failed decision blocks the selected
language's full leaderboard submission.** A valid saved failure is evidence of
incomplete evaluation, not permission to submit it. Keep the original run, diagnose
the cause, and use a fresh run ID for a corrected evaluation. Do not drop the longest
examples, turn capacity failures into abstentions, or invent uniform probabilities.
See [submission eligibility](contributing_results.md#leaderboard-eligibility-comes-before-submission).

## First distinguish the limit that failed

A state that fits by itself may still fail when instructions, authored criteria,
all candidates, special tokens and an answer token are included. Some models also
have separate state/query, candidate-text, head/schema, character or HTTP-byte
limits. Tokens measured with another tokenizer are not evidence that the native
request fits. For a self-hosted service, the running server's per-request/slot
limit matters in addition to the checkpoint's capacity.

| Failure pattern | Implemented examples / evidence | Appropriate response |
| --- | --- | --- |
| Complete request exceeds a configurable serving cap | [TinyJev](../evaluator/src/s1mb/adapters/tinyjev.py), [APUS](../evaluator/src/s1mb/adapters/apus.py) | Increase `--context-limit` only where the adapter and checkpoint permit it; count the complete native request and smoke-check memory. |
| Requested cap exceeds the model/native runtime capacity | [Dinah](../evaluator/src/s1mb/adapters/dinah.py), [Decision 2.0](../evaluator/src/s1mb/adapters/decision2.py), [Julia](../evaluator/src/s1mb/adapters/julia.py) | A larger number is rejected or capped. Restore a supported limit; consider only the adapter's explicitly implemented shortening policy below. |
| Separate query/candidate branch exceeds its cap | [Ichi](../evaluator/src/s1mb/adapters/system_ichi.py), [length checks](../evaluator/src/s1mb/adapters/input_lengths.py) | Identify the overflowing branch. A larger total context or smaller batch does not automatically raise query or candidate limits. |
| Fixed head or candidate text is too long although state fits | [Laya](../evaluator/src/s1mb/adapters/laya.py), [JevK5-Lite](../evaluator/src/s1mb/adapters/jevk5_lite.py), [Lev](../evaluator/src/s1mb/adapters/lev.py) | Check head/schema and per-option limits separately. State-only shortening cannot repair an oversized protected head. |
| Candidate count exceeds the native codebook/head | [APUS](../evaluator/src/s1mb/adapters/apus.py), [Julia](../evaluator/src/s1mb/adapters/julia.py), [RSI](../evaluator/src/s1mb/adapters/rsi.py) | Use an implemented `--max-candidates` extension when supported. Otherwise the question remains unsupported; do not remove or arbitrarily split candidates. |
| JSON bytes, structure or text characters exceed a bound | [FlyMy](../evaluator/src/s1mb/adapters/flymy.py), [OpenAI Decisions](../evaluator/src/s1mb/adapters/openai_decisions.py) | Check transport/structure limits separately from tokens. Use only an existing documented transport extension; increasing a token cap alone is insufficient. |
| GPU OOM or native memory-admission rejection | [candidate batching](../evaluator/src/s1mb/adapters/upstream.py), [LFM-PCD](../evaluator/src/s1mb/adapters/lfm_pcd.py) | Inspect free GPU memory, reduce supported batch/work budgets, and use supported SDPA/FlashAttention. One intact request must still fit. Never fall back to CPU inference. |
| Hosted API rejects a request | [OpenAI Decisions / Luna](../evaluator/src/s1mb/adapters/openai_decisions.py), [TypeSafe](../evaluator/src/s1mb/adapters/typesafe.py) | Identify a permanent input/schema/access rejection versus a transient service failure. Local context flags cannot change a provider's limits. See the Luna procedure below. |
| Too many independent questions in one HTTP request | [GLiDE/OpenRouter transport](../evaluator/src/s1mb/adapters/jev_channels.py) | The implementation chunks independent questions with the same complete state, using 24 questions per request by default. This does not split one question's candidates or shorten its input. |
| Self-hosted server context/slot is too small | [Unee](../evaluator/src/s1mb/adapters/unee.py), [llama.cpp client](../evaluator/src/s1mb/adapters/llama_cpp.py) | Reconfigure the controlled server with sufficient per-slot context, retain overflow rejection, and record its actual runtime configuration. Client settings alone do not resize the server. |
| HTTP success but missing probabilities or malformed output | [OpenAI decoder](../evaluator/src/s1mb/adapters/openai_decisions.py), [probability validation](../evaluator/src/s1mb/data.py) | Preserve failure; inspect the response contract safely. Do not reconstruct missing probabilities from an expected score, or normalize arbitrary invalid output. |

A batch failure is not necessarily an input-length failure. Conversely, reducing
`--case-batch-size` does not reduce the tokens or candidate count of a single
question. `--limit` evaluates fewer cases; it does not make a full benchmark
complete or shorten its longest case. A two-case smoke check can miss every
problematic long input.

## Supported controls and their boundaries

CLI flags are adapter-specific. [Argument validation](../evaluator/src/s1mb/cli.py)
rejects unsupported combinations before inference. Do not copy a capacity flag
from one adapter to another or assume that every accepted `--context-limit`
permits extension beyond the released model capacity.

| Adapter / control | What the current implementation permits | What it does not fix |
| --- | --- | --- |
| `tinyjev`, `nimble`, `mini-jev`, `apus`: `--context-limit` | A full-input cap within the checked backbone position capacity; the original serving/training condition remains relevant. | A candidate codebook limit, another protected field limit, or insufficient GPU memory. |
| `lumma`, `verdict-encoder`, `kotoba`: explicit context extension | Documented larger full-input conditions, including position extrapolation where applicable. Kotoba uses query tiling that retains all keys, not independent document windows. | No retraining or guarantee of long-context quality. See [capacity extensions](../evaluator/OPEN_MODELS.md#explicit-capacity-extensions) for release-specific bounds and provenance. |
| `dinah`: `--context-limit` | A positive limit no larger than 8,192, further capped by the loaded native model. | Raising the native model capacity. |
| `decision2`: `--context-limit` | A positive limit no larger than `engine.max_input_tokens`; native `max_length_exceeded` becomes `Decision2InputTooLong`. | Extending a release beyond its native capacity. |
| `julia`: `--context-limit` | 517–8,192 tokens, with a separate 512-token head and native option constraints. | More than its 20 candidate slots or a generic override of its per-option budget. |
| `lavoir`, `jevk5-lite`: `--context-limit` | A cap up to 1,024 and 512 respectively, also bounded by the loaded release. Lavoir checks its 256-token head independently. | A general long-context extension or removal of fixed-schema limits. |
| `lev`: `--context-limit` | Full request capacity within the checked backbone limit. Mode B separately rejects candidate text over 64 tokens. | Extending that 64-token candidate-text budget. |
| `laya`: `--max-len`, `--head-max-len`, `--questions-per-call` | Explicit total/head packing limits and question grouping. The constructor requires `0 < head_max_len < max_len`, and complete encoding is checked against these caps. | Proof that a raised limit is supported by the backbone, or permission to omit labels. |
| `system-ichi`: `--query-length`, `--document-length` | Separate query and candidate caps, checked with the native tokenizer and special-token accounting before inference. | An automatically enlarged backbone or silent truncation. |
| `apus`, `verdict-encoder`, `winnow` and other CLI-supported adapters: `--max-candidates` | An explicit, model-specific head/codebook extension. APUS verifies distinct single-token codes and requires 16–255; Winnow's documented extension requires its native patch/build. | A universal 255-candidate entitlement. Preserve native computation, all candidates and recorded extrapolation conditions. |
| `bekko-v0`: `--case-batch-size`, `--microbatch-tokens` | Bounded case batching and native work budgets; lower them to reduce memory pressure. | Increasing one request's semantic capacity. Its native adaptive input policy is a separate mechanism. |
| `openai-decisions`: `--case-batch-size` | 1–32 concurrent cases; default 4. Each question is still a separate full-input request. | Provider context/candidate limits, request-byte bounds or policy refusals. |

Some native algorithms already use windows, tournaments or multi-stage readouts.
Preserve those implemented algorithms and their metadata; do not add a new
candidate tournament or document-window average as an unrecorded fallback.
Similarly, JADE's bounded retrieval of candidate token logprobs keeps the same
complete prompt and candidate set; it is not a way to fit an oversized input.

## Explicit shortening is a different evaluation condition

Try a supported lossless serving-cap extension or batching correction first.
When the native capacity is fixed, the following are the implemented alternatives.
They must be selected for technical reasons independently of test scores, recorded
in model settings and the submission, and evaluated under fresh run IDs. They do
not guarantee complete coverage. Preserve original strict measurements and
explicitly disclose any mixture of strict and shortened benchmark results in the
canonical model folder.

| Policy | Supported adapters | Preserved content and remaining failure conditions |
| --- | --- | --- |
| `--state-token-reduction FRACTION` | `dinah`, `decision2`, `jade`; default 0 means strict rejection | Shortens long unprotected body strings using head/tail text and an omission marker. Preserves JSON structure, protected question/instruction/criterion fields and every candidate. The fraction bounds removed rendered-state tokens, not the full context. Protected content that still cannot fit remains a failure. |
| `--native-adaptive-budget` | `julia`, `lavoir` | Uses the implemented native preparation; may shorten state, questions and candidate descriptions, while retaining candidate counts/markers. Hard candidate capacity and an unrepresentable head remain limitations. |
| `--native-adaptive-budget` | `jevk5-lite` | Uses the native body budget; for an oversized non-Score head it compacts redundant anonymous prefixes and shortens long label text, retaining all options in one head and at least 16 original tokens in shortened labels. Complete numeric Score labels remain protected; an oversized Score schema fails. |
| `--native-adaptive-budget` | `kodiak` | Applies native character/token budgets, including head/tail shortening of question/label text without removing labels. This can change authored textual content; it is not the state-only policy. |
| `--native-adaptive-budget` | `saihai-310m` | Uses the released Bekko-style `adaptive-v1` query/candidate preparation; saved model settings record its limits and truncation policy. |
| Native default `adaptive-v1` | `bekko-v0` | Native query/candidate budgeting is already the declared default exception. Record `context_length`, query/candidate budgets and the effective truncation policy; it is not a strict full-input measurement. |

The state-shortening algorithm is implemented in
[`input_budget.py`](../evaluator/src/s1mb/adapters/input_budget.py); adapters measure
their complete native request on each trial. Relevant adapters emit budget logs
with case/question references and before/after lengths; keep these diagnostic
logs outside result roots. Exact behavior of the adaptive alternatives is in
[Julia](../evaluator/src/s1mb/adapters/julia.py),
[Lavoir](../evaluator/src/s1mb/adapters/lavoir.py),
[JevK5-Lite](../evaluator/src/s1mb/adapters/jevk5_lite.py),
[Kodiak](../evaluator/src/s1mb/adapters/kodiak.py),
[Saihai 310M](../evaluator/src/s1mb/adapters/saihai_310m.py) and
[Bekko v0](../evaluator/src/s1mb/adapters/bekko_v0.py).

Do not apply these flags to a hosted API that does not implement them. Do not
rewrite evaluation rows or targets to fit a model, silently shorten instructions
or candidates, or remove hard examples from the active manifest.

## Luna / OpenAI Decisions: when the API rejects an input

This section describes the current
[`openai-decisions` adapter](../evaluator/src/s1mb/adapters/openai_decisions.py).
It does not assert an undocumented Luna context limit or infer a limit from the
model name. Provider rejection can reflect input size, schema, access, safety or
other conditions; an HTTP status alone does not establish which one occurred.

The adapter already sends the complete state with **one question and all of its
options/levels per request**. It uses native typed probabilities, not chat
completions. There is no implemented `--context-limit`, `--max-candidates`,
`--state-token-reduction` or `--native-adaptive-budget` override for this adapter.
Setting `--case-batch-size 1` serializes cases but does not shorten any request.

| Observation | Current client behavior | Action |
| --- | --- | --- |
| Transport failure or HTTP 429, 500, 502, 503, 504 | At most three attempts per `_post`, with 1s then 2s waits. | For sustained rate limiting, reduce case concurrency; for an outage, rerun after service recovery. Retries must retain identical question content/settings. |
| Other non-200 status, including 400, 401, 403, 404, 413 or 422 | No status-based retry; returns a failed prediction. The client retains a status-only error, never an upstream response body. | Check access/model/endpoint and schema first. Confirm whether the provider actually rejected length or candidate capacity before choosing a remedy. Do not repeatedly resend an input known to exceed a fixed limit. |
| HTTP 200 with answer type `refusal` | Decoder records `abstention: "refusal"`. | Keep it in the denominator with the documented incorrect/max-error penalty. This is a scored refusal, not an operational failure; disclose counts by task. Do not paraphrase or retry until an answer appears. |
| Invalid answer count/type/candidate mapping, invalid probabilities, or changed resolved model | Returns a failed prediction; non-runtime input/response errors are intentionally generalized. | Diagnose the contract or model-version change. Fix an actual adapter bug with tests, or pin/record a reproducible provider version when available; rerun rather than rewriting saved predictions. |
| Locally encoded request or streamed response exceeds 8 MiB | The client rejects it with `MAX_BYTES`; this is a client bound, not a published model token limit. | Confirm which bound was reached in a controlled local diagnostic. There is no CLI byte-limit override; a justified bounded-transport change needs code review and validation, not manual edits to result files. |

Use the following procedure for a rejected long input:

1. Identify the affected benchmark/case and inspect the saved settings and failure
   counts. Preserve the original run. Do not assume that a generic `ValueError`
   means token overflow; see the error-recording caveat below.
2. Reproduce the affected complete request with the same model, dataset revision
   and renderer in a controlled adapter diagnostic. Inspect only sanitized
   statuses, error categories and length/count measurements. Calling
   `OpenAIDecisionsAdapter.predict(case.inference())` directly exposes its
   sanitized `Prediction.error` and abstention outcome directly; it still makes real
   API calls, one per question in that case. Keep diagnostics outside results,
   without request/response bodies, credentials or private inputs in public logs.
3. For 429/transient failures, lower supported concurrency or wait for provider
   recovery. For a schema bug, correct the adapter while preserving authored
   meanings, structured values and numeric Score levels, then test the fix.
   The current adapter already isolates questions, so reducing question grouping
   further is not a remedy for Luna.
4. If the provider confirms a hard limit that rejects an intact required question,
   the current API configuration cannot complete that benchmark. A supported
   provider-side capacity increase could be evaluated under a fresh, disclosed
   configuration. Otherwise keep the failure and do not submit the incomplete
   model to the full-scope leaderboard. A proposed new shortening policy requires
   an explicit adapter implementation and review; none is currently available for
   Luna. A timeout/concurrency change cannot repair a hard per-request limit.
5. Rerun affected benchmarks under fresh run IDs after an actual technical fix;
   validate the new measurements and check the complete canonical model folder
   before creating a results PR. Changing to a different model is a separate
   evaluation, not a retry of the same measurement.

Other HTTP adapters have different contracts. TypeSafe and Liquid send their
case's typed questions using their own transport; the shared GLiDE/OpenRouter
client implements independent-question chunking. Do not assume Luna's request
shape, retry status set or limits apply to those adapters.

## Read saved failures and logs correctly

The current [runner](../evaluator/src/s1mb/runner.py) retries an exception from
`predict_batch` case by case. This isolates batch failures; it is not a general
promise to retry each provider rejection. Adapters also have their own bounded
HTTP/native retry policies, so a batch fallback can cause additional calls.

During result collection, the runner preserves valid per-question operational
errors and explicit abstentions, including other valid answers from the same case.
A documented API refusal is an abstention scored as incorrect/max-error, as
specified in the [evaluation guide](evaluation.md#api-refusals-and-unanswerable-judgments).
It is not a missing prediction or a generic HTTP rejection. Operational errors
still make the benchmark incomplete.

If the adapter throws an exception, returns missing/duplicate question IDs, an
invalid case ID or invalid probabilities, the runner marks every question in that
case failed and records only the exception class. A thrown HTTP exception can
therefore become `RuntimeError`; an invalid response can become `ValueError`.
Failed decisions are not the number of failed HTTP requests. Budget or GPU
diagnostic detail may also be absent from saved results.

Use saved `status`, `counts`, predictions, model settings, dataset source SHA and
API usage together with controlled local diagnostics. A run may save partial
benchmark files and then exit nonzero. Initialization errors can instead leave
benchmarks with no result file. Validate files that exist and separately check
active-category coverage. Do not infer a missing benchmark succeeded from the
presence of other saved files.

## Rerun and check submission readiness

All commands below run from `evaluator/`. Keep the same model/dataset pins and
frozen evaluation configuration unless the disclosed technical fix changes one.
Select whole affected benchmarks; `--limit` is a smoke tool, not a repair for
incomplete full coverage. For example, after correcting transient Luna failures:

```sh
uv run --env-file ../.env s1mb run --adapter openai-decisions \
  --model gpt-6-luna --category english-v1 \
  --benchmark AFFECTED_BENCHMARK_ID --case-batch-size 1 \
  --run-id decisions-retry-002
uv run s1mb validate data/results/decisions-retry-002
```

Export the deliberate replacements into a staging copy of the canonical model
folder that also contains its other original measurements. Preserve those
original measurements and explain the retry/configuration change. Then run:

```sh
uv run s1mb validate-results ../tmp/results-submission
uv run python scripts/create_results_pr.py \
  --results-dir ../tmp/results-submission --category english-v1 \
  --explanation-file ../tmp/author-explanation.md \
  --output ../tmp/results-pr-body.md
```

Use `japanese-v1` for a Japanese submission. The PR generator requires all active
benchmarks of the selected language complete and measured with zero failed
decisions, even when `--require-complete` is omitted. If a failure remains, use
`report_results.py` for local diagnosis and continue fixing it; do not upload an
incomplete leaderboard submission. See [result submission](contributing_results.md)
for export, replacement and provenance requirements.
