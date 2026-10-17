# Vision v1 evaluation

The evaluator supports image-bearing Choice, Noul, Score, and Box judgments through the TypeSafe adapter. OpenAI Decisions supports the same image-bearing Choice/Noul/Score tasks, using native predicate for Noul; it does not support Box. Vision is a separate category from the English/Japanese text leaderboards. Box uses normalized full-image left/top/right/bottom coordinates, mean IoU (higher is better), and accuracy at IoU >= 0.5. ScreenSpot converted to Box is not the original point-click protocol.

We gratefully acknowledge [MTEB](https://github.com/embeddings-benchmark/mteb),
[MIEB](https://arxiv.org/abs/2504.10471), and
[MMEB / VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec)
([evaluation dataset](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval)) as
references for candidate discovery and coverage. The original image benchmarks
and image contributors are credited in the [source checklist](evaluation_sources.md#vision-sources)
and [license inventory](vision-source-licenses.md). S1MB's typed adaptations do
not reproduce embedding evaluation protocols or directly comparable scores.

## Choose the measurement scope

`vision-v1` is an independent evaluation scope: 26 benchmarks / 7,800 judgments.
The leaderboard requires all 20 Choice/Noul/Score benchmarks / 6,000 judgments;
the six Box benchmarks / 1,800 judgments are optional. `--all` selects the
English/Japanese text union and does not include vision. Always select
`--category vision-v1` explicitly.

| Adapter | Choice / Noul / Score images | Box | Full-run selection |
| --- | --- | --- | --- |
| `typesafe` | Supported | Supported with Jev 1.14 | All 26 sets, or `--exclude-box` for the required 20 |
| `openai-decisions` | Supported | Unsupported | `--exclude-box` |
| `imajev` | Supported | Unsupported | `--exclude-box --max-images 3`; disclose the experimental three-image extension |

Three-image comparisons are part of the required scope. An adapter that only
supports one or two images cannot complete it by dropping images or assembling a
montage. Image-unaware adapters are rejected. For a new model, follow
[vision adapter development](vision-adapters.md).

## Install and inspect without inference

Run all commands from `evaluator/`. The private
[`hotchpotch/S1MB-vision-v1`](https://huggingface.co/datasets/hotchpotch/S1MB-vision-v1)
release requires an account granted access. Use the existing configuration in
`dataset-source.json`; no local category, copied definitions, or manually
constructed `vision-cases.json` is needed for a normal evaluation.

```sh
uv sync --locked
uv run hf auth login
uv run python scripts/fetch_dataset.py --category vision-v1
uv run s1mb list --category vision-v1
uv run s1mb check-data --category vision-v1
```

Fetching and structure validation do not invoke a model. Acquisition verifies
the release and atomically installs cases and image blobs. English, Japanese and
vision caches coexist under the normal `data/` root. `list`, `check-data` and
`validate` operate locally and do not fetch missing data.

Normal `run` commands resolve the configured Hub revision once before model
loading and record its exact SHA in results. A failed online check stops the run.
Use `--offline-dataset` only to deliberately reuse the installed release. For a
reproducible Dataset pin, set the `vision-v1` source revision in
`dataset-source.json` to its full SHA; `run --revision` selects the model revision.

## Smoke check, then full evaluation

Configure provider credentials using [`.env.sample`](../.env.sample). Before paid
inference, estimate the selected work and obtain any applicable budget approval;
the project requires explicit approval when expected cost reaches 1 USD. Count
image inputs, question requests and possible retries, not just benchmark files.
For CUDA models, inspect free GPU memory first, install the model's documented
extras, and pin checkpoint/source revisions; do not fall back to CPU inference.

This Jev smoke checks three cases **per benchmark**, including Box (78 judgments).
Use a fresh run ID for every invocation:

```sh
uv run --env-file ../.env s1mb run \
  --adapter typesafe --model jev-1.14.0 --category vision-v1 \
  --limit 3 --run-id jev-vision-smoke-UNIQUE
uv run s1mb validate data/results/jev-vision-smoke-UNIQUE
```

Inspect failed counts, image receipts, model settings, dataset repository/SHA and
safe diagnostics before proceeding. Validation checks saved measurements; a
successful limited run remains partial. Include a targeted `--benchmark
nights-choice` smoke when checking a new adapter's three-image support. Smoke
checks may fix technical problems, but must not select settings by test scores.

After fixing the configuration, remove `--limit` and use a new ID:

```sh
uv run --env-file ../.env s1mb run \
  --adapter typesafe --model jev-1.14.0 --category vision-v1 \
  --run-id jev-vision-full-UNIQUE
uv run s1mb validate data/results/jev-vision-full-UNIQUE
```

For Decisions, use `--adapter openai-decisions --model gpt-6-luna --exclude-box`
in both smoke and full commands. For Imajev, use the pinned CUDA invocation
[below](#private-hub-release-and-leaderboard), adding `--limit 3` for its smoke.
`--task`, `--benchmark`, and `--generalization-only` are useful diagnostics but do
not establish full leaderboard coverage. Preserve failed runs and rerun with a
fresh ID after technical fixes.

## Validate and prepare a submission

Create display metadata following the [submission guide](contributing_results.md#prepare-a-submission),
then export and run the vision completion gate before uploading:

```sh
uv run s1mb export-results data/results/jev-vision-full-UNIQUE \
  --metadata ../tmp/my-model/metadata.json --output ../tmp/vision-submission
uv run s1mb validate-results ../tmp/vision-submission
uv run python scripts/create_results_pr.py \
  --results-dir ../tmp/vision-submission --category vision-v1 \
  --output ../tmp/vision-pr-body.md
```

Complete the generated explanation and disclosure fields. The gate requires all
20 non-Box benchmarks with zero operational failures; validation of only the
files present does not prove completion. Optional Box results retain their own
coverage. Follow the [Dataset PR workflow](contributing_results.md#open-a-hugging-face-dataset-pr)
to review and upload the package, then Publish the finished PR and verify it is
Open. Keep adapter/code changes in a separate source PR. Do not upload private
images, inputs, targets, local reports or credentials.

## Local installation format

A development data root uses the existing `categories/` and `benchmarks/`
definitions plus `dataset-source.json`. A benchmark dataset directory can contain
`vision-cases.json`, a JSON array of `Case` records. Each case includes `images`,
mapping neutral names such as `image_1` to
`{ "path": "../image-cache/<sha256>", "sha256": "<64 lowercase hex digits>", "policy": "scene-896" }`.
Paths must remain inside the common dataset storage directory. Use `state.images`
for the ordered neutral names, and those names in instructions/state. Targets and
provenance remain separate. All image bytes are checksum-verified before inference.
Use global `--data-dir PATH` consistently for acquisition, checks, runs and
validation; results then default to `PATH/results/<run-id>/`.

Keep original compressed image bytes in the canonical dataset (for example,
shared Parquet Image blobs referenced by SHA-256). Extracting an identical-byte
local read cache is optional. `vision-cases.json` is the evaluator installation
format, not a replacement for source/license manifests. Its
`provenance.input_hash` must bind the authored input, image hashes and preprocessing
policy. Keep development data roots ignored by Git.

## Provider preparation

Jev accepts PNG/JPEG with each side between 280 and 1568 pixels and rejects out-of-range images without automatic resizing. Requests use multipart files and the documented `TypeSafe-Danger-Zone-Images: true` header. Box requires Jev 1.14; pin `jev-1.14.0` rather than assuming an alias resolves to a compatible model. Each image is inserted once in state under a neutral UUID reference; questions refer to the state path. Source identifiers, file paths, hashes, targets, and annotation metadata are not sent.

Three explicit policies preserve the whole image:

- `scene-896`: uniform scaling to a maximum side of 896, with a minimum side of 280 where possible. Suitable as a starting point for general scene/relation/semantic tasks.
- `detail-1568`: maximum side 1568 for documents, GUI, counting, and small defects. It does not enlarge already sufficient originals.
- `original`: preserve exact upright PNG/JPEG bytes and resolution. Use this for image-quality targets tied to the original rating resolution; reject incompatible originals rather than silently change their quality.

For extreme aspect ratios, scale uniformly and add centered white padding to satisfy the minimum side. No cropping or ground-truth-dependent preprocessing is used. Box predictions are mapped from the padded content rectangle back to original normalized coordinates. A prediction wholly in padding maps to a zero-area box and receives zero IoU. Non-original policies use lossless PNG transport; TIFF documents are converted. Image receipts retain source/payload hashes, dimensions, padding rectangle, byte size, and policy in result environment metadata.

These sizes are starting policies, not proven optima. Compare resolution variants on fixed cases and report actual response usage; the provider has not published an image-token estimation formula. Images count as input tokens. Smaller file bytes reduce transfer/storage but do not by themselves establish token savings.

## Score compatibility

The Jev API accepts at most 10 Score levels. The adapter rejects wider scales rather than changing the requested primitive. TallyQA uses exactly ten numeric levels, 0 through 9. Restrict the eligible author-test questions to original answers in that range before drawing 300 cases. Preserve exact counts and record the excluded rows and sampling frame; do not clip counts, merge higher answers into 9, or select a different scale per question. This is a restricted-count subset, not the complete original TallyQA protocol.

See [source licenses](vision-source-licenses.md) before distributing any data. Current API references: [images](https://docs.typesafe.ai/images), [Box](https://docs.typesafe.ai/primitives/box), [API/Score](https://docs.typesafe.ai/api), and [models/pricing](https://docs.typesafe.ai/models).

## OpenAI Decisions images

Use `--adapter openai-decisions --model gpt-6-luna` for the native Decisions endpoint. Select only supported benchmarks using repeated `--benchmark` flags, or select one of `--task choice`, `--task noul`, and `--task score`. Keep Box tasks explicitly outside that measurement scope; they are unsupported, not zero-scoring predictions. Do not substitute another model or generate coordinates through a different endpoint.

The adapter sends the authored state as text, followed by each neutral image label and its `input_image` part in the declared order. Images use base64 data URLs and explicit `detail: high`. It uses the same full-image preparation policies as TypeSafe and records payload hashes/dimensions so comparisons can verify identical uploaded bytes. Provider-side image processing and tokenization may still differ. Original IQA bytes remain unchanged locally. Requests contain one complete question each; images are inserted once per request. The existing 8 MiB client request bound includes base64 expansion. All original criteria, numeric Score levels, native probabilities, refusal handling, and no-truncation policies remain in effect.

See the [official OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions) and [input image schema](https://developers.openai.com/api/reference/resources/decisions/methods/create).

## Imajev

The CUDA adapter accepts up to two ordered images using the shared full-image
policies, followed by the pinned model's native multimodal processor. It retains
the existing one-rotation calibrated readout and conditional unknown policy.
Score criteria use the adapter's numeric Choice fields; this is distinct from
the upstream HTTP Ordinal transport. Box is unsupported. Three-image comparison
tasks require explicit `--max-images 3` with `--adapter imajev`. This native
processor extension exceeds the upstream two-image serving contract and is
recorded as `experimental-three-image-extension` in model settings. Images retain
their original order and shared preprocessing policy; there is no montage or
image removal. The processed-token limit still rejects overflow. Record the source, adapter and base revisions; inspect training
exposure before interpreting results as held-out performance.

Zero-area predicted boxes are retained verbatim and receive zero IoU. Ground-truth
boxes must have positive area. Inverted, non-finite or out-of-range coordinates
remain invalid. This accounts for collapsed edges in quantized provider output
without widening the box, dropping the example or selectively retrying it.

## Vision v1 general tasks

The `vision-v1` scope contains 18 source-derived benchmarks and 8 synthetic
English general benchmarks (26 sets / 7,800 judgments total). General IDs are
`generalization-vision-{easy|hard}-{noul|choice|score|box}`, each 300 cases:
150 WIT images and 150 images from the earlier source pool. `--generalization-only`
selects those 8 sets (2,400 judgments). The category is `vision-v1`.

The general cases preserve authored English instructions, state, all answer-space
contract fields, option order and numeric Score anchors. Noul maps yes/no to
true/false without changing their authored definitions. Confidence and review
reasons stay outside installed cases. Teacher labels, source information and
image identifiers remain outside model-facing state. Source groups, individual
licenses, fixed permalinks and original split lineage remain in provenance.
Synthetic evaluation rows use the local evaluator's `test` slot; original source
partitions are preserved and are not asserted to be author-held-out tests.

General images retain original compressed bytes in the local checksum-verified
read cache. Three WIT assets are multi-image MPO still photographs; installation
extracts their primary frame 0 (the frame used for authoring), applies EXIF
orientation and stores a lossless RGB PNG derivative. Original blobs remain in
the cache/export; per-case derivation receipts bind both hashes and frame index.
No alternate stereo view is added to model input. `detail-1568` is the explicit full-image inference policy, preserving
small Box targets and applying the shared padding/inverse-coordinate rules.
Teachers were authored using EXIF-oriented images capped at 1280 px; inference
preprocessing is a separately recorded policy, not identical authoring pixels.
Labels were authored with GPT-6.1 Sol and checked by separate answer-blind calls
from the same model family, plus subagent spot checks. They are synthetic
teachers, not human gold. Easy/hard is a reviewed design classification rather
than measured performance. Neither unseen training exposure nor unrestricted
redistribution is established. The Hub release is private; this does not establish redistribution clearance.

Offline structure validation (no inference/API charges):

```bash
uv run s1mb check-data --category vision-v1
```

Paid evaluation requires an explicitly approved cost estimate whenever the
expected total reaches 1 USD. Installation and structure validation do not grant
permission to perform inference.


## Private Hub release and leaderboard

`hotchpotch/S1MB-vision-v1` contains 26 configurations / 7,800 judgments, each with
300 cases. Twenty non-Box configurations / 6,000 judgments are required for ranking.
Six Box configurations / 1,800 judgments are optional. Use `--exclude-box` for
models without native Box support:

```sh
uv run --env-file ../.env s1mb run --category vision-v1 --adapter imajev --model mohit67890/imajev-4b \
  --revision MODEL_SHA --source /path/to/pinned/imajev --device cuda:0 \
  --max-images 3 --exclude-box --run-id fresh-vision-run
uv run --env-file ../.env s1mb run --category vision-v1 --adapter openai-decisions --model gpt-6-luna \
  --exclude-box --run-id fresh-luna-vision-run
```

Calls to paid providers need a cost estimate and any applicable budget approval
before execution. `--all` retains its existing English/Japanese text union;
select `vision-v1` explicitly for visual models.

The release stores exact Case records in a JSON-encoded Parquet column and
SHA-256-deduplicated original compressed assets in separate Parquet shards.
`checksums.json` binds every release file. The installer verifies all shards,
case schemas, image hashes, active manifest membership and counts before an atomic
installation. Inputs/targets remain separate. Prediction-only result validation
can download the pinned case subset without downloading image pixels.

The viewer offers English v1, Japanese v1 and Vision v1. Vision Task Avg and Borda
require all non-Box benchmarks, with no Box weight. Independent Box / General Box
columns report mean IoU ×100, requiring complete coverage of the respective Box
subset only. Missing or unsupported values display as unavailable, never zero.
Jev can contribute Box; Luna and imajev can submit the complete non-Box scope.
Submission reports include both optional columns while enforcing non-Box completion.
Training-exposure metadata accepts all vision benchmark IDs through the same
`training_exposure` schema as text: known source family and actual evaluation-example
use remain distinct. Unknown exposure is never displayed as unseen.
