# S1MB evaluator

Python package and CLI for typed Choice, Noul, and Score evaluation. Run commands
from this directory using Python 3.11 and uv.

## Install and evaluate

```sh
uv sync --locked
uv run s1mb list
uv run s1mb run --adapter dummy --category smoke-v1 --limit 2 --run-id smoke
```

The dummy adapter checks plumbing with synthetic predictions. `smoke-v1` selects
one benchmark per task; `english-v1` selects the complete evaluation set. Omit
`--limit` for full evaluation. A limited run is always marked partial. Each run
requires a fresh `--run-id`; saved runs are not overwritten.

`data/results/<run-id>/` holds predictions, recomputed metrics, model configuration,
timing, input hashes, and `environment.dataset_source` with the exact dataset
repository and commit SHA. Results are local and ignored by Git. Reports and
non-result JSON must stay outside result directories.

## Dataset acquisition

`s1mb run` checks the latest `main` revision of
[hotchpotch/s1mb-dataset](https://huggingface.co/datasets/hotchpotch/s1mb-dataset)
once before loading the model. `dataset-source.json` selects the repository and
requested revision; use a full commit SHA to reproduce a specific release.

An unchanged installed revision uses local Arrow files without subset API calls.
New downloads are checksum-verified and converted from Parquet without changing
row contents. The original source files remain alongside the local materialization.
`data/datasets/hub-source.json` records the resolved SHA. A lock prevents another
CLI evaluation or refresh from replacing the dataset during a run.

The repository currently requires Hugging Face read access. Use `hf auth login`
or `HF_TOKEN`; credentials are never stored in result files. A failed online
check stops evaluation. `--offline-dataset` explicitly uses installed data without
a network request. `list`, `check-data`, and `validate` always operate locally.
An optional explicit refresh is available as `uv run python scripts/fetch_dataset.py`.

Only the active evaluation manifest defines membership. Quarantined data never
enters categories. Changes to existing rows or benchmark membership require review;
the importer will not silently reinterpret old results against different data.
The supported row format has `input`, `targets`, `split`, `case_id`, `group_id`,
`language`, and `input_hash`. Criteria order and numeric values are preserved.
Only decoded input is sent to models; targets and provenance remain evaluator-side.

## Real adapters

Use `uv run s1mb run --help` for the current options. Upstream dependencies are
loaded only for the selected adapter. Model checkpoints and external source
checkouts are not part of this repository.

### TypeSafe / Jev

Set `TYPESAFE_API_KEY`, then select the API model:

```sh
uv run --env-file ../.env s1mb run --adapter typesafe --model jev \
  --category smoke-v1 --limit 2 --run-id jev-smoke
```

API calls may incur charges. After inspecting smoke results, use `english-v1`
without `--limit` and select a new run ID.

### Bekko

Use an environment containing the checkpoint's supported Bekko package,
Sentence Transformers, PyTorch and FlashAttention dependencies. Point `--source`
to the upstream checkout and `--model` to the checkpoint directory.

```sh
CUDA_VISIBLE_DEVICES=1 python -m s1mb run \
  --adapter bekko --source /path/to/bekko-system-one --model /path/to/model \
  --device cuda --query-length 16384 --document-length 2048 \
  --microbatch-tokens 64000 --category smoke-v1 --limit 2 --run-id bekko-smoke
```

Bekko uses its current native training renderer, optimized inference, batched
probability transfers, and batched input-length validation. The default token
budget is 64,000; lower it when GPU memory is limited. Complete decisions stay
together in a microbatch. Explicit branch limits reject overflow rather than
silently truncating. Effective settings are recorded in every result.

### Other adapters

`uv sync --locked --extra laya` installs Laya dependencies. Other local adapters
use separate upstream source checkouts and supported model environments; see
[adapter notes](OPEN_MODELS.md) and `upstream-models.json`. Dataset-default
instructions are used throughout. Pass GPU and model settings explicitly and
inspect partial runs or failures before comparing scores.

## Validate and develop

```sh
uv run s1mb check-data --category english-v1
uv run s1mb validate data/results
uv run tox
```

`validate` requires at least one result. Do not run it on an empty checkout.
Public unit tests require no dataset credentials or GPU. Dataset integration
tests skip explicitly when the release is absent; use `S1MB_TEST_NO_DATASET=1`
to reproduce that configuration. Shared fixtures keep Python and viewer scoring
consistent. See [SCORING.md](SCORING.md) for metric definitions.

### Parameter metadata

Local adapters record `model.total_params`, `model.active_params`, and
`model.parameter_count_method` in each new result. API models and older results
with unavailable counts use null/absent values, never an estimated zero.
The viewer displays both counts in the run's model identity details.

`non_lookup_parameters_v1` uses the mmBERT embedding project's non-lookup AP
convention: count all unique registered parameters (including frozen weights and
task heads), then subtract lookup-only `Embedding` and `EmbeddingBag` weights.
Shared weights are counted once; an embedding tied to an output projection remains
active. Position/type lookup tables are also excluded. Buffers are excluded from
both counts. This is not per-token MoE routing, FLOPs, or trainable parameter count.

To calculate the same metadata independently, provide an importable Python factory
returning the **complete** model or native runtime wrapper, including task heads:

```sh
uv run python scripts/count_parameters.py my_model:load --kwargs '{"checkpoint":"/path/to/checkpoint"}'
# Equivalent module entry point:
uv run python -m s1mb.parameters my_model:load --kwargs '{"checkpoint":"/path/to/checkpoint"}'
```

For a local Transformers sequence-classification checkpoint, its standard loader
can be used directly (after checking physical GPU 1's free memory):

```sh
CUDA_VISIBLE_DEVICES=1 uv run --extra open-models python scripts/count_parameters.py \
  transformers:AutoModelForSequenceClassification.from_pretrained \
  --kwargs '{"pretrained_model_name_or_path":"/path/to/checkpoint","local_files_only":true,"device_map":"cuda:0"}'
```

Use the checkpoint's actual model class; an encoder-only loader would omit task
heads and produce a different total.

Install the factory's model dependencies first. It may build an architecture on
the meta device to avoid allocating weights, provided it preserves the checkpoint's
architecture and tied parameters. The counter runs no inference and makes no device
transfers or network requests. If the factory loads weights on this workspace, use
physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`) and check free memory first. Existing
measurements are not rewritten; new evaluations count the actual loaded model.
