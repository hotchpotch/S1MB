# Adding a model to the S1MB leaderboard

This guide covers submission rules, result packaging, Hugging Face Dataset pull
requests, and leaderboard updates. Follow [Running evaluations](evaluation.md)
first for environment setup, smoke checks, full runs, and result validation.

Evaluation results belong in the Hugging Face results dataset. Changes to adapters,
scoring code, or documentation belong in a separate source-code PR. Link that PR
when a submission depends on code that has not been merged yet.

`ORG/RESULTS` below is a placeholder, not a provisioned repository. Use the results
repository designated by the leaderboard maintainer; do not upload to the
benchmark input dataset configured in `evaluator/dataset-source.json`.

All Python/Hub CLI commands below run from `evaluator/` unless stated otherwise.

## Submission rules

- Submit actual model measurements. The dummy adapter produces demo results and
  cannot represent a measured leaderboard entry.
- Aim for complete `english-v1` coverage across Choice, Noul, and Score. Partial
  submissions may add benchmark results, but must disclose missing or failed
  benchmarks and do not qualify for aggregates requiring complete coverage.
- Use dataset-default instructions and the model's supported adapter. Preserve
  authored criteria, structured input, numeric Score levels, and soft targets.
  Reject input overflow; do not silently truncate or drop difficult examples.
- Record the evaluated model/checkpoint, source revision, effective settings, and
  evaluator version. Every result must contain the evaluation dataset repo ID
  and exact commit SHA. Do not manually invent or replace measurement provenance.
- Different dataset revisions, evaluator versions, and original run IDs may be
  combined in a model folder. Results are validated and scored against their
  recorded dataset revision. A revision difference alone is not disqualifying.
- Add missing benchmark files or replace existing ones. Explain replacements in
  the PR, including the changed configuration or reason for rerunning. Do not
  assemble an undisclosed best-of-many-configurations row.
- Use a different model folder when versions or configurations should appear as
  separate leaderboard entries. Use fresh local run IDs for reruns; replacing
  a published benchmark does not require overwriting the original local run.
- Disclose known evaluation/training overlap and non-default inference behavior.
  S1MB is a mosaic of specialized tasks; its scores do not establish unseen-task
  generalization or training-data non-overlap.

The format and score checks establish consistency, not the identity of the model
that generated predictions. Maintainers review the submitted evidence and may
request corrections or a reproducible rerun before merging.

## Repository layout

The Hugging Face results dataset is a file repository with one folder per
leaderboard entry. It does not require Parquet or a generated index:

```text
typesafe__jev_1_14/
  metadata.json
  <benchmark-id>.json.xz
hotchpotch__bekko_17M_v0/
  metadata.json
  <benchmark-id>.json.xz
```

Use `<organization-or-user>__<model-id>` for the folder name. A new model version
can use a new ID. Add missing benchmark files or overwrite an existing benchmark;
the Hub commit history preserves earlier versions. Different original run IDs,
evaluator versions, and dataset revisions may coexist in one model folder.
Each file keeps its original measurement provenance. One folder is one leaderboard
row, regardless of the original run IDs.

## Model metadata

```json
{
  "model_id": "example__model_v1",
  "display_name": "Example Model v1",
  "short_name": "Example v1",
  "url": "https://example.org/model",
  "hf_url": null,
  "total_params": null,
  "active_params": null,
  "parameter_count_method": null
}
```

The three ID/name fields are required. Links must be HTTP(S); links and parameter
counts may be omitted or null. Active parameter counts require
`parameter_count_method: "non_lookup_parameters_v1"`: exclude lookup-only
embeddings, retaining shared output weights. Total counts may be unknown.
When either metadata count is supplied, the metadata counts take precedence for
display, including unknown values. Otherwise counts come from measured results.
Measurement files are never rewritten to change display names. Use consistent
counts within a model folder, or declare them in metadata.

## Prepare a submission

Create `../tmp/my-model/metadata.json` using the schema above, substituting your
model ID and names. For the commands below, its ID is `example__model_v1`.
Keep metadata drafts, staging folders and PR notes under ignored `tmp/`.

After following the [evaluation guide](evaluation.md), export the selected result
files or run directory. Replace `MY_RUN` with the completed local run ID:

```sh
uv run s1mb export-results data/results/MY_RUN \
  --metadata ../tmp/my-model/metadata.json \
  --output ../tmp/results-submission

uv run s1mb validate-results ../tmp/results-submission
```

The exporter validates predictions, metrics, counts and input hashes before
writing files. It preserves original run IDs, model revisions, dataset repo IDs
and exact dataset commit SHAs. New evaluations also record the evaluator Git SHA
when available. Existing evaluations retain their recorded evaluator version.
Missing dataset SHAs are rejected; do not guess them or fill them in by hand.
The exporter adds/replaces the specified benchmarks, preserving other files.
Submitting multiple different results for the same benchmark in one export is
ambiguous and rejected; select one result explicitly.

If you already have packaged results from an older dataset revision, fetch the
recorded data and validate that package with:

```sh
uv run s1mb validate-results /path/to/already-packaged-results --download-datasets
```

That command expects a repository of model folders, not a raw run directory. For
a raw run, retain its original evaluation data or install its recorded dataset
revision in a separate data root, as described in the [evaluation guide](evaluation.md#dataset-revisions).

Inspect the staged model folder. It must contain only `metadata.json` and the
intended `<benchmark-id>.json.xz` files. `validate-results` takes the repository
root (`../tmp/results-submission`), not the individual model folder. It validates
files that are present; successful validation does not imply complete coverage.
Each JSON file must decompress to at most 64 MiB.

## Open a Hugging Face Dataset PR

Authenticate with an account allowed to access the designated results repository:

```sh
uv run hf auth login
```

Prepare the PR description using the template below. Then upload only the model
folder, excluding the export directory's local lock files:

```sh
uv run hf upload ORG/RESULTS \
  ../tmp/results-submission/example__model_v1 example__model_v1 \
  --repo-type dataset --create-pr \
  --commit-message 'Add Example Model v1 results'
```

The upload creates a Dataset PR rather than updating `main`. Open the returned
PR, or find it under the dataset's Community tab. Add the completed description
and inspect the file diff: every changed path should be inside the intended model
folder, and replacements should be deliberate. A compressed diff is not enough
for review; include the validation and coverage summary in the PR body.

To update the same PR after review, use its existing ref. Replace `12` with its
PR number; do not pass `--create-pr` again:

```sh
uv run hf upload ORG/RESULTS \
  ../tmp/results-submission/example__model_v1 example__model_v1 \
  --repo-type dataset --revision refs/pr/12 \
  --commit-message 'Update Example Model v1 results'
```

Uploading a folder adds or replaces those files; it does not remove other
benchmarks already in the repository. Do not use deletion flags to clear a model
folder when merely adding coverage. If upload fails because of repository access,
resolve that access with the maintainer rather than publishing to a different repo.

See the [Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)
for the underlying Dataset PR workflow.

### PR description template

Copy this into the PR and replace each placeholder. Record facts from the saved
results and execution logs; do not infer missing settings from model names.

```markdown
## Model

- Leaderboard folder: `example__model_v1`
- Model/checkpoint and model card URL:
- Exact model revision, or API-resolved version:
- Adapter and upstream source revision, if applicable:
- Evaluator version / Git commit (include uncommitted changes, if any):

## Evaluation

- Exact evaluation command(s):
- Dataset repo ID and commit SHA(s):
- Coverage: completed / required benchmarks for Choice, Noul, and Score:
- Missing, partial, or failed benchmarks and reasons:
- Machine/runtime: GPU, memory, OS, Python, relevant package versions:
- Effective precision, attention backend, batching and input limits:
- Non-default inference behavior and known training/evaluation overlap:
- Reruns, retries, or configuration differences within this folder:

## Results and changes

- Added benchmark files:
- Replaced benchmark files and reasons:
- Metadata changes:
- Validation command and outcome:
- Complete overview scores, if available (baseline-adjusted, 0–100):
- Relevant raw task metrics; do not report partial coverage as an overall score:
```

Overview scores use baseline adjustment per benchmark, clipping, averaging within
each task, and equal weighting of the three tasks. Noul uses balanced accuracy
for adjustment; its primary raw metric is Brier, where lower is better. See
[the scoring specification](../evaluator/SCORING.md) before quoting scores.

### Maintainer review

Before merging, check the model ID and metadata, the intended additions and
replacements, coverage, and reproducibility information. Validate the PR revision
using the preview command below. Review changes to inference code separately from
result data. Partial files must remain marked partial, and demo results must
remain demo results.

For initial setup, create the dataset repository separately and configure `*.xz`
as large files in its `.gitattributes`. Its README should describe this layout, the evaluation
command/revision information expected in PRs, and the applicable results license.
The code's MIT license does not grant dataset or checkpoint redistribution rights.
Review files before publishing: a public results repository exposes predictions
and recorded environment/settings, even though the viewer sends only summaries.
Do not submit credentials, raw input text, checkpoints, caches, or reports.

## Synchronize the leaderboard

```sh
uv run s1mb sync-results --repo-id ORG/RESULTS
```

The command resolves `main` to one exact commit, downloads model metadata and
compressed results, and validates all files before atomically switching the local
`data/hub-results` symlink. It keeps previous local snapshots; a failed download
or validation leaves the previous snapshot installed and exits with an error.
The resolved results SHA is printed and appears in the snapshot directory name.
Use `--output` to choose another local destination. An existing ordinary directory
is not replaced by synchronization.

To review an unmerged PR, use a separate destination:

```sh
uv run s1mb sync-results --repo-id ORG/RESULTS \
  --revision refs/pr/12 --output ../tmp/results-pr-12
```

Results are checked against their recorded evaluation dataset SHA. The installed
evaluation dataset is reused when it matches; other revisions are downloaded by
exact SHA and materialized under ignored `data/result-datasets/`. Only the needed
subsets are materialized, with Hub checksum and active-manifest checks. The
current benchmark/category definitions control leaderboard membership. Counts,
inputs and targets may differ between releases; task, dataset, split and primary
metric must still identify the same benchmark. Scores and baselines are computed
against each result's own release before aggregation. Unknown or incompatible
benchmark IDs are rejected.

For a locally downloaded repository, prepare and validate its dataset revisions:

```sh
uv run s1mb validate-results /path/to/results --download-datasets
```

From `viewer/`, restart with `npm start`. When `data/hub-results` exists, it is the
default result source; otherwise local `data/results` is used. Explicit
`--results-dir /path/to/results` selects another source. The viewer needs the
`xz` executable (`xz-utils` on Debian/Ubuntu); decompressed JSON is limited to
64 MiB per file. No network requests are made by the viewer. Public builds and
unit tests need no Hub account, private data, model, or GPU.

## After merge

Merging a Dataset PR does not itself restart the application. Run `sync-results`
and restart the viewer after merging, manually or in the deployment's scheduled
job. The viewer exposes per-benchmark dataset/model/evaluator revisions in model
details. Missing or partial benchmarks remain visible and cannot claim complete
aggregate coverage. Results remain self-reported: consistency checks do not prove
which model generated a prediction.
