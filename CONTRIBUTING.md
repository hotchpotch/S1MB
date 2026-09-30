# Contributing

For leaderboard submissions, start with [Running evaluations](docs/evaluation.md)
and [Adding a model to the leaderboard](docs/contributing_results.md). Submit
result files through a Hugging Face Dataset PR; keep adapter and other code
changes in a separate source-code PR.

Use Python 3.11 and uv under `evaluator/`, and Node.js 22.22.2/npm under `viewer/`.
Follow the README commands and directory-specific AGENTS.md instructions.

Keep changes focused and include tests for input boundaries, scoring changes,
result validation, and model-adapter behavior. Run Python tests, lint and type
checks with `uv run tox`; run viewer tests, typecheck and build for viewer changes.
The public CI configuration uses synthetic fixtures and needs no private dataset
access. Dataset integration tests are clearly marked and run locally when data
is installed. Do not substitute demo results for measured model performance.

Do not commit credentials, downloaded datasets, model weights, predictions,
benchmark reports or local audit output. Store such artifacts in ignored local
directories. Source dataset and model licenses remain separate from this project's
MIT license. Include the applicable attribution when adding third-party material.

Support the current dataset schema and inference interfaces. Do not add migration
shims, historical format readers, or temporary experiment switches. Record effective
model configuration and dataset revision for reproducibility instead.

Describe the problem, resulting behavior and validation in a pull request. Network
publication, model API calls, and GPU measurements should be explicit; unit tests
must not perform them implicitly.

See [adapter development](docs/adapters.md) for the inference contract and tests,
and [viewer setup](docs/viewer.md) for local UI development. GitHub provides a
bug report form, a [model evaluation request form](docs/model_requests.md), and a
source PR template. Model requests are voluntary suggestions, not promised work.
