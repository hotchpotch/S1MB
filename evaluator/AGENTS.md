# Evaluator guidance

Run Python 3.11/uv commands from this directory. Use `uv sync --locked` and run
`uv run tox` after implementation changes. Tests, Ruff and type checks must pass.
Public CI runs without downloaded data; dataset integration tests use the
`dataset` marker and skip when the evaluation release is absent. Do not fetch data
or call model APIs from unit tests.

Follow `../docs/evaluation.md` and `../docs/contributing_results.md`; update them
when CLI behavior changes. After evaluation-data changes, run
`uv run s1mb check-data --category english-v1`. Validate changed raw runs with
`uv run s1mb validate data/results/RUN_ID` against their evaluation data. Validate
published model folders with `uv run s1mb validate-results REPOSITORY_ROOT`;
add `--download-datasets` only when recorded revisions need acquisition.
Validation is not required on an empty public checkout.

Local runs require fresh IDs. Exported benchmark files may be added or replaced;
do not apply raw-run identity restrictions to a published model folder. Keep
`metadata.json` separate from measurements and preserve recorded provenance.
Use bounded XZ decoding and atomic writes. Synchronize to a verified snapshot
before switching the active result source; do not silently retain stale data
while reporting a successful refresh.

Adapters implement the common interface under `src/s1mb/adapters/`. Use only the
current compact dataset schema and native model input conventions. Bekko defaults
to batched transfers and a 64,000-token microbatch budget; users can lower the
budget for available memory. Preserve candidate alignment. Bekko v0 uses the Hub's remote
`BekkoSentenceTransformer.predict()` API with code and weights pinned to the same
SHA. Do not maintain a local Bekko inference backend. It uses
`adaptive-v1` input budgeting with documented truncation instead of overflow
rejection; other adapters retain their declared overflow policies.

Standard output is `data/results/<run-id>/`. Keep downloaded data, results, Hub
snapshots, revision caches, audits, and explicit `output/` directories ignored.
Do not commit them or include them in package artifacts. Keep scoring aligned with shared viewer fixtures.
Write documentation, comments and docstrings in English.
