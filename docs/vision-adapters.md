# Developing a vision adapter

Start with the common [adapter contract](adapters.md) and the
[vision evaluation workflow](vision-evaluation.md). A vision adapter implements
the same `predict`, `metadata`, and `close` methods as a text adapter, and declares
`supports_images = True`. This attribute enables image evaluation in the runner;
it does not provide image transport, preprocessing or native model support.

## Working implementation examples

| Native interface | Implementation | Offline tests |
| --- | --- | --- |
| Multipart typed API; Choice, Noul, Score, Box | [`TypeSafeAdapter`](../evaluator/src/s1mb/adapters/typesafe.py) | [`test_vision.py`](../evaluator/tests/test_vision.py) |
| Ordered image parts and native per-question Decisions | [`OpenAIDecisionsAdapter`](../evaluator/src/s1mb/adapters/openai_decisions.py) | [`test_openai_decisions.py`](../evaluator/tests/test_openai_decisions.py) |
| CUDA multimodal processor and calibrated decision readout | [`ImajevAdapter`](../evaluator/src/s1mb/adapters/imajev.py) | [`test_imajev_vision.py`](../evaluator/tests/test_imajev_vision.py) |

Use the example matching the model's native interface. Do not copy a provider's
probability rounding tolerance, Noul encoding, Score ordering or calibration to
a model with different semantics. See the current source for its enforced
limits and transformations. An adapter must reject unsupported tasks and image
counts explicitly. Box support is optional for the vision leaderboard; three-image
support is needed to complete the required non-Box benchmarks.

## Image input boundary

The runner passes an `InferenceCase`, containing authored state, questions and an
ordered `images` mapping of neutral names to `ImageAsset` records. Targets and
provenance are excluded. Image paths and checksums are evaluator-side information:
never serialize `case.model_dump()` or the asset records into model input.
Keep the neutral names associated with their corresponding pixels in insertion
order. Preserve the authored state and questions while translating image references
into the model's native representation.

The following helper is a reusable boundary example, not a complete adapter or
a new provider API. It prepares ordered bytes for a native renderer and retains
the receipts expected by the runner:

```python
from s1mb.data import InferenceCase
from s1mb.vision import prepare_image


def prepare_case_images(
    case: InferenceCase,
    image_receipts: dict[str, list[dict]],
    *,
    max_images: int,
) -> list[tuple[str, bytes, str]]:
    if not case.images:
        raise ValueError("Vision inference requires images")
    if len(case.images) > max_images:
        raise ValueError("Native image-count limit exceeded")
    parts = []
    receipts = []
    for name, asset in case.images.items():
        payload, mime, receipt = prepare_image(asset)
        parts.append((name, payload, mime))
        receipts.append(receipt)
    image_receipts[case.case_id] = receipts
    return parts
```

Initialize `self.image_receipts = {}` on the adapter and call this helper with
that mapping. The runner saves receipts for each benchmark's selected cases in
`environment.image_preprocessing`. Send only the neutral names, payload bytes
and MIME types to the native image renderer. For an in-process model, decode the
prepared bytes into independent PIL images as Imajev does; then apply the pinned
model's native processor. Record any additional processor resizing or tiling in
model settings. Check the complete processed input length before inference,
including image tokens, and reject overflow without dropping or cropping images.

`prepare_image` verifies source checksums and applies the case's declared policy.
It preserves the full image, with padding where required. `original` preserves
upright PNG/JPEG bytes for image-quality judgments and rejects incompatible
sources. Do not replace a declared policy after inspecting scores. Source/payload
hashes, dimensions and the padding rectangle remain in receipts, outside model
text. Do not silently add another lossy encoding step.

## Return native probabilities or original-image boxes

For Choice/Noul/Score, return one `Prediction` per question with probabilities
keyed by the original option IDs. Preserve structured descriptions, authored
Noul outcomes, declared option order and numeric Score values. Native transport
keys may be anonymous; map them back before returning. Keep IDs out of semantic
model text and retain them only for transport/output matching.

Do not turn a generated answer, expected ordinal value or ranking into a
fabricated probability distribution. Unsupported tasks, malformed native
responses and inference failures remain errors. See the common contract for
explicit model abstentions and bounded batching.

For a single-image native Box model returning normalized coordinates on the
prepared payload, undo the shared padding/scale transform before returning:

```python
from s1mb.data import Box, Prediction
from s1mb.vision import original_box


def decode_native_box(case, question, native_edges, receipt):
    if question.task != "box" or len(case.images) != 1:
        raise ValueError("This Box decoder requires one image and a Box question")
    payload_box = Box.model_validate(native_edges)
    return Prediction(
        case_id=case.case_id,
        question_id=question.id,
        box=original_box(payload_box, receipt),
    )
```

Here `native_edges` has `left`, `top`, `right`, `bottom` in [0, 1]. Convert native
pixel or center/size outputs to this convention first. The receipt must belong
to that same image. If the native processor adds another geometric transform,
invert it before calling `original_box`; do not apply the shared inversion twice
to outputs already expressed in original-image coordinates. Predictions wholly
in padding become zero-area boxes and score zero IoU. Preserve valid zero-area
predictions; reject inverted, non-finite and out-of-range native boxes. No target
box is needed for decoding.

## Integrate and verify

Register the adapter through the CLI's existing lazy construction path, as
described in [adapter development](adapters.md). Document supported image counts,
tasks, full installation and invocation, pinned model/source/base revisions,
precision, attention backend, preprocessing, probability origin and overflow
policy. Reject incompatible options before inference where possible. Do not
advertise Box support through `supports_images`; it only declares image input.

Use generated images, fake native processors and mocked HTTP responses in public
tests. The existing vision tests demonstrate these checks without private inputs,
API credentials, downloads or a GPU:

- One-, two- and three-image order; each neutral reference receives the correct
  pixels, with no source IDs, hashes, paths or targets in model-facing content.
- Shared checksum validation, original-byte preservation, elongated-image padding,
  Box coordinate round trips, zero-area predictions and invalid coordinates.
- Native task limits, unsupported Box, image-count and processed-token overflow,
  option alignment, numeric Score mapping and malformed responses.
- Metadata and receipts, bounded failures, cleanup and saved-result validation.

From `evaluator/`, run `uv sync --locked` and `uv run tox` for an implementation
change. Then follow the [smoke/full workflow](vision-evaluation.md#smoke-check-then-full-evaluation).
A real smoke checks integration and capacity; it must not tune on test scores.
Keep local measurements ignored, submit the implementation as a source PR, and
use a separate validated results Dataset PR for measurements.

## Additional local decision models

The following bridges use released native visual backbones and probability
readouts. Pin weights and runtime code independently using
[`upstream-models.json`](../evaluator/upstream-models.json). These bridges support
Choice, Noul and Score; use `--exclude-box` for the required vision scope. None
claims native Box support. The shared full-image policy runs before each model's
own processor; different native pixel budgets remain part of the model protocol.

| Adapter | Checkpoint family | Native readout and input details |
| --- | --- | --- |
| `onejev` | OmniJev OneJev 1.6, 0.8B / 4B / 9B | Native multimodal engine, ordered PIL media, released calibration, six-decimal response rounding |
| `vev` | CountingSheep Vev 4B / 9B | Native state image objects and allowed-token softmax; weights are **CC BY-NC 4.0**, independently of Apache-2.0 runtime code |
| `vision-jev` | LMDHQ Vision-Jev 0.8B / 9B | Released LoRA, typed heads and temperatures; authored criteria replace the runtime's fixed image-quality Score rubric |
| `jevvision` | divyanshx11 JEVision | Released visual sidecar and pointer head; never substitutes the text route |
| `visual-jev` | guanxuyu Visual Jev 4B answer-SFT | Native answer-letter readout; Noul and numeric Score are transported as complete supplied choices; native 200,704-pixel budget |
| `decider-vision` | Mapika Decider 2B Vision | Native answer-slot logits; explicit 255-label table prevents the image helper's default ten-option selection |
| `clef` | Cloudflare Clef-Flash | Native joint-schema encoder receives the processor and all ordered PIL images |
| `pplx-decider` | Perplexity Decider Flash v1.2 | Native joint protocol, released temperature applied once, native noncausal full-attention masks preserved |

OneJev, Vev, JEVision, Clef and Perplexity accept multiple images in their native
interfaces. Vision-Jev, Visual Jev and Decider Vision extend their single-image
example renderers with ordered image placeholders and the full native multimodal
processor. This is an input transport extension, not additional training. Record
it when interpreting multiple-image tasks. Never concatenate images into a
montage or select a subset to fit a single-image interface.

Noul retains both authored definitions. Score retains every original numeric
value inside the visible criterion and maps the resulting probabilities back to
the original option order. A Choice transport does not establish ordinal
calibration. Structured instructions are retained as complete JSON text when a
runtime requires strings. Inputs exceeding the configured token budget are
rejected; criteria and state are never truncated.

For a three-case smoke check with a locally cloned runtime:

```bash
CUDA_VISIBLE_DEVICES=0 uv run s1mb run --adapter onejev \
  --model OmniJev/OneJev1.6-0.8B \
  --revision 0acd65660ac22ca42f40b8b6a6da3a039590b412 \
  --source /path/to/pinned/OneJev --device cuda:0 \
  --category vision-v1 --exclude-box --limit 3 --run-id onejev-vision-smoke
```

Use a fresh run ID and omit `--limit` only after every selected benchmark passes.
`jevvision` and `pplx-decider` load their runtime from the pinned checkpoint bundle
and do not require an external `--source`. Vision-Jev additionally requires
`--subfolder vision-jev-0.8b` or `vision-jev-9b` and an explicit `--base-revision`;
Visual Jev also requires `--base-revision`. The registry records reproducible base
revisions. Clef uses its existing `--adapter clef` command and bundled runtime.

Review training exposure separately from accuracy. In particular, Vision-Jev's
KonIQ normalizer independently re-splits the original dataset; its candidate train
pool can overlap author-test images. A source-family disclosure does not certify
that evaluation examples were excluded. Missing disclosure means unknown.

### Text-only evaluation of visual checkpoints

JEVision is a routed bundle: its declared default `text/jevvision-text` adapter
handles text-only inputs, and its visual sidecar handles image-bearing inputs.
The bridge follows this native routing, retaining the same pinned public bundle,
strict authored criteria, numeric Score levels, and the text branch's 80,000-token
ceiling. It never uses a fabricated image or substitutes the visual head for the
released text branch. Metadata identifies the selected text component and native
Kev settings. The image-bearing route remains unchanged.

Visual Jev and Vision-Jev keep the original structured state for text-only
requests. Visual Jev passes no media to the processor and removes the native
image marker. Its answer-SFT readout remains limited to 2–16 native answer tokens.
English questions with more candidates remain unsupported; do not split their
candidate sets, invent predictions, or claim complete English coverage. Complete
Japanese coverage can be submitted independently when all its benchmarks pass.

## Additional released local runtimes

`d3` runs the Decision 3.0 checkpoint's bundled `d3_runtime.D3` in-process on CUDA,
with its released BF16 backbone, FP32 readout, calibration and full-input overflow
rejection. Its native Choice/Noul/Score protocol retains authored binary criteria
and numeric Score anchors. Use a full checkpoint SHA and `--context-limit 32768`.
The checkpoint bundle is the default `--source`.

For larger D3 checkpoints, `--cuda-shards 2 --device cuda:0` places complete
decoder layers across exactly two visible CUDA devices. Weights are staged on
CPU during loading, then moved entirely to CUDA before inference. Placement
reserves activation memory and rejects checkpoints that cannot fit; it does not
offload inference to CPU. Record both device IDs through `CUDA_VISIBLE_DEVICES`.
OneJev also accepts `--cuda-shards 2` for compatible Qwen vision checkpoints.
Its native FP32 output head remains on GPU 0 and is included in the available
memory calculation before balancing complete backbone layers. Native calibration,
slot readout and sequential fork execution are preserved.
`pplx-decider --cuda-shards 2` uses the same CUDA-only backbone placement.
Its native CPU loader initially stages FP32 weights, so host memory requirements
are higher than the BF16 checkpoint size. Before any inference, both backbone
and readout are converted to the native CUDA BF16 dtype and moved to CUDA.
`rsi --cuda-shards 2` supports the native separate vision/scoring architecture:
vision modules, scoring heads, calibration buffers and the final norm stay on
GPU 0, while complete text decoder layers are split between both GPUs. Adaptive
policy calibrators are recreated on CUDA from the same release files after
loading; the released default effort policy is retained.
For inference-only final-layer heads, the two-GPU adapter retains only the final
and declared auxiliary-exit hidden states. Long eager requests therefore release
unused intermediate activations instead of retaining every decoder layer. Native
readout indices, final normalization, calibration and inputs remain unchanged;
other readout configurations are rejected. Results record this retention policy.

`rsi` also accepts ordered images with a compatible RSI-Jev VL checkpoint. Pin
both the model and the RSI-Jev Git checkout (`--source`). Image requests use the
native full-depth vision path; text requests retain the released effort policy.
The existing explicit Choice transport preserves authored Noul and numeric Score
criteria. The published model card's Apache-2.0 weight declaration must be read
alongside its separately disclosed non-commercial/research-only training sources.

`image-hopper` uses the checkpoint's bundled Image Hopper predictor, pinned Qwen
base, released LoRA routing and default serving options (including a 768-token
image cap on its default route). It requires images and has no text-only fallback.
Noul and Score use native Choice with all authored criteria and numeric anchors;
this transport uses the release's route/option-count temperatures. The native limit is 26
options and four images. The checkpoint bundle is the default `--source`.

All three adapters apply the shared benchmark image policy first and retain
receipts for the subsequent native preprocessing. Box is unsupported. Model
cards and source-family training disclosures remain separate from exact-example
contamination claims.

`intern-decision` also accepts ordered images with a compatible native checkpoint
and Python 3.12 or newer. The bridge applies the benchmark image policy, then
passes lossless temporary PNGs with neutral numbered names to the native image
processor. Paths and source metadata are excluded from model text. It preserves
authored Noul definitions and transports Score anchors as numeric-keyed criteria.
Its native limit is 62 candidates; larger questions are unsupported and cannot
be reduced to claim complete scope coverage. Box is unsupported.

`amos` loads a released Amos decoder bundle (`model/config.json` and `serve/amos`)
and calls its native in-process engine. It preserves released per-type calibration,
one option order, native image preprocessing and complete-input overflow rejection.
Use `--cuda-shards 2` for large Qwen vision checkpoints. The output head remains on
GPU 0 and complete backbone layers are balanced across both CUDA devices after
CPU loading. Noul keeps authored criteria; Score retains numeric level descriptions
and its full probability vector. Legacy Amos encoder checkpoints and Box are
unsupported by this bridge. Run a fresh image smoke before full evaluation.

`e12ex-vision` loads the released e12ex2 vision LoRA and its explicitly pinned
Qwen base. Supply `--source` as a pinned Decision Index kit checkout,
`--base-revision` and `--cuda-shards 2`. The checkpoint supplies the native
`code/serve/engine.py` runtime. Weights are staged on CPU, positioned entirely on
CUDA, then merged on CUDA before the native option-logit readout runs. One option
order and the release's identity temperature are retained. All typed questions
use explicit native Choice criteria because the native binary shortcut ignores
authored definitions; numeric Score anchors remain in their descriptions.
The released label table supports 624 candidates, with the first 62 corresponding
to the training format. Inputs are rejected on overflow, and Box is unsupported.

```sh
uv run s1mb run --category vision-v1 --exclude-box --adapter d3 \
  --model vllm-sr/d3-mini --revision 97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e \
  --device cuda:0 --context-limit 32768 --limit 3
```

## Additional public-weight bridges

A bridge's presence does not establish a completed benchmark result. Check the
submitted coverage, settings, source hashes and failures.

| Adapter | Runtime and conditions | Image limit |
| --- | --- | --- |
| `matilda` | Bundled calibrated decision runtime; optional two-GPU CUDA placement | Native processor |
| `startlux` | Released Transformers pipeline for Instant checkpoints, including bidirectional parity checks and per-type temperatures | Native pipeline |
| `rune` | Released decisions scaffold and complete candidate logits on Transformers; optional two-GPU CUDA placement with 5 GiB per device reserved for long-input activations | Native processor |
| `winnow --winnow-vision` | Verified Q8 GGUF and matching projector through the released CUDA server; direct decisions | 16 |
| `span` | Released Rust/CUDA engine and deterministic adaptive policy; requires its pinned MTP assistant through `--base-revision` | 4 |
| `xor-vision` | Released two-order wrapper with a local Transformers candidate-logprob bridge | Native processor |
| `autotrust-vision` | Published raw System 1 choice scaffold, PEFT adapter, trained choice bias and released choice temperature | Native processor |
| `clef-omni` | Released joint-schema runtime, unused speech output disabled, CUDA-only thinker placement | Native encoder |
| `neohorse` | Bundled `VisionDecisionEngine`, preserving image/token guards | 1 |
| `jev-omni` | Bundled image-file runtime with a temporary lossless PNG | 1 |
| `shisa-vision` | Released SDK image scaffold with complete candidate logits; direct image read, uncalibrated as in the SDK | 1 |

AutoTrust and Shisa transport Noul and Score through their native Choice
scaffolds to retain all authored outcome descriptions and numeric Score values.
Shisa's Noul object otherwise substitutes fixed Yes/No descriptions. This
mapping is recorded in settings. `shisa-vision` uses the SDK's direct image read,
which differs from the Decision Index's repeated-input policy. The Transformers
protocol bridges differ from their authors' SGLang/vLLM serving backends; retain
these distinctions when reporting measurements.

Without images, `shisa-vision` uses the SDK's direct DE-2 text scaffold and full
256-slot codebook. This permits wide text choices without dropping candidates;
the SDK's single-image path retains its separate 26-option limit.

Single-image runtimes cannot complete NIGHTS or NLVR2. Keep those tasks marked
unsupported, retain all original images, and report supported-task measurements
as partial coverage. They do not qualify for the complete-suite ranking.

Use `--cuda-shards 2` for large bridges where supported. CUDA placement reserves
activation memory and rejects CPU/disk offload. CPU staging while loading weights
is distinct from CPU inference. Span's 9 GB KV pool is smaller than the author's
large-GPU submission pool; its deterministic policy is retained, and capacity
failures remain failures. Keep native binary hashes, source revisions, model
revisions and the assistant revision in run settings.
