# Developing a model adapter

An adapter translates S1MB inference inputs into a model's native interface and
returns typed probabilities. Start with [evaluation](evaluation.md) for runtime
setup and [code contributions](../CONTRIBUTING.md) for source PR requirements.
Existing runtime-specific instructions remain in [adapter notes](../evaluator/OPEN_MODELS.md).

## Interface and source map

- [`adapters/base.py`](../evaluator/src/s1mb/adapters/base.py): `ModelAdapter`
  protocol and API encoding/decoding helpers.
- [`data.py`](../evaluator/src/s1mb/data.py): `InferenceCase`, `Question`,
  `Prediction`, `ModelInfo`, and probability validation.
- [`dummy.py`](../evaluator/src/s1mb/adapters/dummy.py): minimal implementation;
  its uniform predictions are demos, not model measurements.
- [`cli.py`](../evaluator/src/s1mb/cli.py): adapter selection, options, lazy imports,
  construction, and cleanup.
- [`runner.py`](../evaluator/src/s1mb/runner.py): bounded batching, failure handling,
  result provenance, and scoring.

Implement these methods in `evaluator/src/s1mb/adapters/your_model.py`:

```python
from s1mb.data import InferenceCase, ModelInfo, Prediction

class YourAdapter:
    def predict(self, case: InferenceCase) -> list[Prediction]:
        # Encode state/questions, call the model, and map probabilities to IDs.
        raise NotImplementedError

    def metadata(self) -> ModelInfo:
        # Return the resolved model revision and effective runtime settings.
        raise NotImplementedError

    def close(self) -> None:
        # Release any model resources or API clients owned by this adapter.
        pass
```

This is an interface outline, not an inference implementation. Add the adapter
name, lazy import and construction to the CLI; validate required and unsupported
options. Put optional runtime dependencies in the appropriate extra in
`evaluator/pyproject.toml` and update the lockfile. Importing the CLI or running
public tests must not load a checkpoint or require that extra.

## Preserve the task meaning

The runner supplies `InferenceCase`, which excludes targets and provenance.
Use `case.state` and the authored questions. IDs exist for matching outputs;
keep case/question identifiers and provenance out of model text. Use anonymous
choice keys when native API transport requires keys that would otherwise reveal
source labels. Do not feed an entire evaluation `Case` to a model.

Preserve dataset-default instructions, structured values, declared criterion
order, and the authored Noul true/false meanings. Score options have numeric
values: ranking distributions are not Score labels. If a native API requires a
different ordering, map responses back to the original option IDs. The helpers
in `base.py` support some native interfaces; use them only when their semantics
match the model.

Return one `Prediction` per question with matching `case_id` and `question_id`.
Probabilities must have exactly the declared option IDs, be finite and within
[0, 1], and sum to one within the validator's tolerance. For Noul, return both
`false` and `true` probabilities. A prediction contains either probabilities or
an error, never both. Do not manufacture uniform answers on inference failure.

Check input lengths before inference and reject overflow. Bekko v0 is the
explicit exception: its native adaptive budgeting and truncation policy are
recorded in metadata. Do not introduce silent truncation for another adapter.
Record actual precision, attention backend, input limits, batching, model/source
revision and transformations in `ModelInfo`; omit secrets. Do not guess revisions
from display names. Define credential variable names only in [`.env.sample`](../.env.sample).

## Batching and failures

Optionally expose `case_batch_size` and
`predict_batch(cases: list[InferenceCase]) -> list[list[Prediction]]`.
Return one prediction list per case in input order and use bounded batches.
The runner retries a failed batch case by case, so `predict` must work on its own;
API adapters should account for possible repeated calls and their cost.
Preserve failures in saved results rather than dropping difficult cases.

## Verify and submit

Add synthetic tests under `evaluator/tests/`, using fake model/API responses.
Cover all supported tasks, nontrivial option order, structured values, authored
Noul definitions, numeric Score levels, malformed probabilities, overflow and
failure behavior. If batching is supported, verify case alignment and failure
fallback. Test metadata and cleanup without downloading weights or calling APIs.
Existing `test_typesafe.py` and adapter-specific tests provide examples.

From `evaluator/`, run `uv sync --locked` and `uv run tox`. Then follow the
[evaluation guide](evaluation.md) for a real-model smoke check before a full run.
On the shared workspace use physical GPU 1 after inspecting free memory; do not
silently fall back to CPU. Keep measurements and reports outside source control.

Document installation, supported tasks, exact invocation and inference limitations
in the adapter notes. For a new adapter, we recommend opening its GitHub source PR
at the same time as the separate [HF Dataset PR](contributing_results.md) and
linking the two. Include the adapter, tests and setup instructions so others can
reproduce the evaluation in a different environment without your local code.
