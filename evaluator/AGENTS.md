# Evaluator guidance

Run Python 3.11/uv commands from this directory. Use `uv sync --locked` and run
`uv run tox` after implementation changes. Tests, Ruff and type checks must pass.
Public CI sets `S1MB_TEST_NO_DATASET=1`; dataset integration tests use the `dataset`
marker and skip when the evaluation release is absent. Do not fetch data or call
model APIs from unit tests.

After data or measured-result changes, run `uv run s1mb check-data --category
english-v1` and `uv run s1mb validate data/results` when results exist. Result
validation is not required on an empty public checkout.

Adapters implement the common interface under `src/s1mb/adapters/`. Use only the
current compact dataset schema and native model input conventions. Bekko defaults
to batched transfers and a 64,000-token microbatch budget; users can lower the
budget for available memory. Preserve overflow rejection and candidate alignment.

Store local data/results/audits in ignored directories. Do not commit them or
include them in package artifacts. Keep scoring aligned with shared viewer fixtures.
Write documentation, comments and docstrings in English.
