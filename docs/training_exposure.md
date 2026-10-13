# Disclosing training exposure

S1MB shows an icon beside leaderboard model names with declared exposure in the
selected scope. Hover or focus it for the meaning, domains and reasons. Click a
model name or icon to open its
sidebar, then expand `Known domains (N)` and a domain to inspect its task, reason
and pinned evidence. The count is the number of distinct disclosed benchmarks in
the selected language scope. Comparisons show compact icons with hover/focus
tooltips beside model names and matching benchmark results.
These annotations do not change AVG, Borda, eligibility, measurements, or rankings.
Missing metadata means **unknown / not disclosed**, never "unseen" or "not trained".
Training disclosures concern the submitted model/configuration; they are not an audit
of unknown base-model pretraining.

## Metadata format

Add `training_exposure` to the model folder's `metadata.json` when training used a
benchmark's source family, related domain, or evaluation examples. Keep the actual
training provenance and disclosure together. Omit the field or use `null` when no
verified declaration is available; an empty array is rejected to avoid implying
independence. The UI identifies these as submitter declarations.

Each entry targets one stable `benchmark_id` and identifies the training Dataset
repo, exact 40-character lowercase commit `revision`, `config`, `split`, a pinned
HTTP(S) `evidence_url`, and a short explanatory `note`. Use multiple entries if
several training sources apply. Do not include credentials, raw examples, gold,
predictions, local paths, or private records. A private Dataset may be named without
redistributing it; evidence links still require the appropriate access.

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
revisions, credential-bearing URLs and invalid relationship values are rejected.
Every benchmark ID must refer to a S1MB definition; retired definitions may remain
in historical metadata, but only active benchmarks affect ranking.

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
