# Disclosing training exposure

S1MB shows an icon beside leaderboard model names with declared exposure in the
selected scope. Hover or focus it for the meaning, domains and reasons. Click a
model name or icon to open its
sidebar, then expand `Known domains (N)` and a domain to inspect its task, reason
and pinned evidence. The tooltip and details show the number of distinct disclosed
benchmarks over all benchmarks in the selected language scope, for example `1/137`.
Multiple training sources for one benchmark count once. The mark communicates
documented training exposure, not an explanation or prediction of high scores.
Other models may have trained on the same domains even when their public
information does not confirm it. Comparisons show compact icons with hover/focus
tooltips beside model names and matching benchmark results.
These annotations do not change AVG, Borda, eligibility, measurements, or rankings.
Missing metadata means **unknown / not disclosed**, never "unseen" or "not trained".
Training disclosures concern the submitted model/configuration; they are not an audit
of unknown base-model pretraining.

**Known domain** means disclosed training on the same source family or a related
domain. It does not establish that S1MB evaluation examples were used or that
record-level overlap was independently audited. Exact training revisions and
splits may be undisclosed; inspect each entry's evidence and limitations. A pinned
manifest can document retained training source names and counts without making
the underlying training rows available for an independent overlap audit. Actual
evaluation-example use is labeled separately as **Evaluation examples used**.
The sidebar explains this distinction below **Known domains**.

## Metadata format

Add `training_exposure` to the model folder's `metadata.json` when training used a
benchmark's source family, related domain, or evaluation examples. Keep the actual
training provenance and disclosure together. Omit the field or use `null` when no
verified declaration is available; an empty array is rejected to avoid implying
independence. The UI identifies these as submitter declarations.

Each entry targets one stable `benchmark_id` and records the training `dataset`
repo, exact 40-character lowercase commit `revision`, `config`, and `split` when known, a pinned
HTTP(S) `evidence_url`, and a short explanatory `note`. Use multiple entries if
several training sources apply. Do not include credentials, raw examples, gold,
predictions, local paths, or private records. A private Dataset may be named without
redistributing it; evidence links still require the appropriate access.
Prefer commit-pinned release files. When the author disclosed training only in a
submission discussion, link to that identified comment, record the retrieval date
in the note, and preserve the retrieved declaration and its digest in the local
audit. A discussion comment remains editable and is weaker archival evidence than
an immutable release file; do not describe it as a commit-pinned training receipt.

| `relationship` | Meaning | Viewer label |
| --- | --- | --- |
| `same_source_family` | Training examples from the same source dataset/family, possibly another task or split | Known domain |
| `related_domain` | Related subject/task domain with documented evidence, without established same-source identity | Known domain |
| `evaluation_examples` | Actual S1MB evaluation examples were used as training inputs | Evaluation examples used |

Do not classify ordinary same-source train examples as direct evaluation-example
exposure without evidence. Conversely, if exact evaluation examples were trained
on, declare `evaluation_examples`. Say whether the overlap assertion comes from a
maintainer report or an independent record-level audit. Domain similarity alone
cannot establish exact overlap. A split named `train` does not establish separation
from every S1MB source: follow the benchmark's original split lineage and adopted
exceptions in [evaluation](evaluation.md) and [source partitions](language_scopes.md#scores-and-source-partitions).

The schema permits at most 256 entries; IDs are bounded to 128 characters, config to
128, split to 64, URLs to 2,048, and notes to 1,000. Unexpected keys, unpinned
non-null revisions, credential-bearing URLs and invalid relationship values are rejected.
A verified declaration may omit `dataset`, `revision`, `config`, or `split`, or set them to
`null`, when actual training snapshot or source details are unavailable. Keep its
primary evidence pinned and explain the limitation in `note`. Never substitute a
current Dataset SHA for an unknown historical training revision. Unknown source
details do not imply an unseen domain. If the assembled training corpus has no
published repository, leave `dataset` null and name the verified source families
and effective retained counts in `note`; do not invent a repository identifier.

The authored registry uses `{ "disclosures": [{ "model_ids": [...],
"training_exposure": [...] }] }`. It accepts at most 256 groups, each with 1–256
exact model row IDs and 1–256 exposures. Duplicate model IDs within or across groups
are rejected. Explicit model-folder disclosures take precedence.

Every benchmark ID must refer to a S1MB definition; retired definitions may remain
in historical metadata, but only active benchmarks affect ranking.

## Reviewing English model disclosures

Apply the same relationships to English v1 as to Japanese v1. Review all published
model folders at one recorded results Dataset SHA, including incomplete rows.
Check submission PRs, the measured model revision's card, release provenance,
training manifests, and linked training code or technical reports. A current card
may describe newer weights: verify release continuity before applying it to an
older measured configuration. Record immutable evidence links for each conclusion.

Use scores only to prioritize this review. A useful exploratory comparison uses
the six `s1mb-generalization-*` benchmarks separately from the other English
benchmarks, adjusts and clips each benchmark using the normal reference baseline,
then averages equally across the three tasks within each group. Compare models
with similar active parameter counts and require coverage of every benchmark in
both groups. Flag an unusually high score outside the generalization group for
research; never turn a score difference into a training disclosure. These groups
have different difficulties, and neither proves unseen domains or non-overlap.

A positive annotation requires evidence that the submitted release actually used
the source family. Dataset access, a supported loader, an evaluation table, or a
collection of eligible training configs does not establish use. Follow effective
license exclusions, filtering, sampling, and conversion receipts. Keep unresolved
sources and inaccessible records unknown. If retained source names/counts are
verified but the assembled corpus is unpublished, annotate the supported family
with null source details and state the retained source evidence in the note.

Keep curated declarations in the authored registry and model-folder metadata;
leave measurement files unchanged. Validate schemas and benchmark references,
prepare the full public snapshot offline, and compare its scores, counts and
coverage with the original snapshot. Publish metadata through a results Dataset
PR and regenerate the display artifact after compatible viewer code is deployed.
The public leaderboard does not refresh from a changed local registry.

## Training for English decisions

The public [Bekko System One v0 training Dataset](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0)
contains source families represented in `english-v1`. A model that actually trained
on these families needs the same disclosure as a model trained on Saihai's Japanese
families. Use `same_source_family` for source-family training, even when another
split or decision task was used. Use `evaluation_examples` when actual evaluation
inputs were training inputs. The Dataset's availability alone does not establish
that another model used it. A large score advantage or a gap between specialized
and synthetic tasks is a research lead, not evidence sufficient for a disclosure.

For each submission, record the actual training revision, configs and splits,
pinned evidence, and any effective filtering. The Dataset includes test configs;
using its train split does not mean using every published split. Source-specific
licenses and access terms still apply. Synthetic tasks described as generalization
probes do not establish unseen-task generalization or training-data non-overlap.

## Bekko v0 example

The pinned release cards for
[17M](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m/blob/b886a1f9b91f4e8368d7830080d52d955e2c8dfa/README.md),
[68M](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m/blob/ab7685f23e5edbc1acb12ced4f2c4e12591efa69/README.md), and
[400M](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m/blob/1960df5602bd93cc8d336fdebd9fb68a30926e13/README.md)
record training on
`hotchpotch/bekko-system-one-dataset-v0@c6a49c450fe88b40bd96c54d9cc19fafd9aa5591`.
They report a full capped-pool training budget with a 300,000-judgment cap per
subset. The [actual training manifest](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0/blob/c6a49c450fe88b40bd96c54d9cc19fafd9aa5591/training-manifest.json)
contains 153 train configs. The direct config-name intersection with English S1MB
source subsets is 77, mapping to 98 benchmark definitions: 49 Choice, 40 Noul and
9 Score. The benchmark-level mapping is maintained in
[`evaluator/model-training-exposure.json`](../evaluator/model-training-exposure.json).
This direct mapping is not an exhaustive inventory of base-model pretraining,
renamed upstream sources, or related domains.

The later Dataset revision `9c673a09c31e145684ab9915871afa72ba40a383` has the same
train config names but removes two complete train cases containing evaluation
inputs. Its [pinned decontamination receipt](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0/blob/9c673a09c31e145684ab9915871afa72ba40a383/decontamination-20261008.json)
reports whitespace-normalized full-input duplicates for these benchmarks:

| S1MB benchmark | Original training config | Exposure established by the receipt |
| --- | --- | --- |
| `contract-nli-choice-test-v1` | `contract_nli` | One evaluation input with 17 decisions |
| `laya--enron-spam-noul-test-v1` | `laya__enron_spam` | One evaluation input with one decision |

The original training revision's Parquet files were checked against the receipt's
before-deletion SHA256 digests, and both reported training case IDs remain present
in that revision. Combined with the release's reported full-pool training, this
supports `evaluation_examples` for these two benchmarks. The other 96 direct
mappings remain `same_source_family`. This is artifact verification of the
maintainer's targeted duplicate report, not a comprehensive independent overlap
audit. Matching inputs does not assert matching targets: the Dataset card also
reports an upstream ContractNLI train/test target disagreement.

Deleting examples from a later training Dataset does not retroactively change
released weights. Disclosures for these Bekko releases must retain their actual
training SHA, rather than substituting the later corrected Dataset revision.
Future models using the corrected release should declare only the families and
examples actually retained in their training; do not inherit the old
`evaluation_examples` declarations without evidence.

## Training for Japanese decisions

To strengthen a model's Japanese typed-decision capabilities, use the
[Saihai training Dataset](https://huggingface.co/datasets/hotchpotch/saihai-v1-dataset)
for training and development, subject to its access requirements and source-specific
licenses. Its dedicated Japanese source-family configurations include training
examples from ten dataset families represented in `S1MB-japanese-v1`. The maintainer
reports removing duplicate examples and overlaps with the adopted evaluation
examples during preparation. This is a maintainer statement, not an independent
record-level overlap audit or a claim that the evaluation domains are unseen.

For an S1MB submission, declare each applicable benchmark as `same_source_family`
in the model folder's `metadata.json` under `training_exposure`; the viewer shows
these entries as **Known domains**. Record the exact training Dataset revision,
configurations, splits, pinned evidence and effective filtering. Declare only the
source families actually used. Deduplication does not remove the disclosure
requirement. If actual evaluation examples were used, declare
`evaluation_examples` instead. Keep S1MB evaluation targets out of training and
parameter selection; select settings using separate training or validation data.
See the [submission rules](contributing_results.md#submission-rules).

The release-specific table below provides the source-family mapping for Saihai's
pinned training release; it is not a declaration for every model using this Dataset.

## Saihai 8B example

The evaluated **Saihai 8B release / Transformers BF16** pins its training Dataset to
`hotchpotch/saihai-v1-dataset@906504ac4b72e133a613011318afd8206fb5e8ec`.
Its [release training provenance](https://huggingface.co/hotchpotch/saihai-v1-8b/blob/798638665cc835e21f84e970531c8565043ce098/training-provenance.json)
records 33,532 train rows and reports that evaluation judgments were not training
inputs. The [pinned Dataset manifest](https://huggingface.co/datasets/hotchpotch/saihai-v1-dataset/blob/906504ac4b72e133a613011318afd8206fb5e8ec/dataset-manifest.json)
includes the source-family configurations below. Use its exact config IDs when
authoring metadata; do not rename IDs from the pinned manifest. Because the release used the whole
train split, these ten benchmark source families are declared known domains.
This is source-family evidence plus a maintainer statement, not an independent
record-level audit, a complete pretraining inventory, or a claim of unseen-domain
generalization. Dataset/model/evidence repositories remain private.

| S1MB benchmark | Relationship |
| --- | --- |
| `commonsensemoralja-noul` | Same source family |
| `jamp-choice` | Same source family |
| `janli-choice` | Same source family |
| `jcola-noul` | Same source family |
| `jcommonsenseqa-choice` | Same source family |
| `jnli-choice` | Same source family |
| `jsick-sts-score` | Same source family; NLI training → STS evaluation |
| `jsts-score` | Same source family |
| `kuci-choice` | Same source family |
| `mintaka-ja-boolean-noul` | Same source family |

For example, if another model actually used this pinned Dataset's whole train
split, declare these same source-family entries. If it filtered out source families,
use the *effective* training inventory/filter receipt and declare only retained
families. Merely having access to this Dataset is not evidence it was used.
Synthetic Japanese tasks and unrelated training configs do not automatically prove
exposure to every S1MB benchmark; add `related_domain` entries only with an explicit,
supported domain mapping.

Use the exact training config and pinned evidence from the release manifest in
each disclosure entry; include only the source families actually used.

See [result contribution](contributing_results.md) for validation and review. The
publisher keeps this metadata separate from saved measurements and carries it into
the bounded offline viewer summary. It never places exposure labels in model inputs
or fetches training data at runtime.
