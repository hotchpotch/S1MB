# External model adapters

S1MB keeps each external inference implementation in a separate checkout.
`upstream-models.json` lists source repositories and known checkpoint references;
these are configuration references, not benchmark results or endorsements.
Install the dependencies required by the selected upstream model separately.
No upstream code or weights are vendored into the evaluator.

Use `--source /path/to/checkout`, an explicit `--model` and `--device`, and run
`smoke-v1 --limit 2` before a complete evaluation. Record distinct configurations
under distinct run IDs. Inspect the result's model metadata for resolved weights,
source hashes, rendering, attention backend, precision and input limits.

| Adapter | Interface |
| --- | --- |
| Bekko | Native typed training renderer and optimized shared-prefix inference |
| System Ichi | Local typed model with bounded question batching |
| Laya | Explicit checkpoint with dataset-default typed instructions |
| Von | Native option-marker backend and calibration |
| JevForge | Candidate-path inference followed by a joint softmax |
| Kev | Native typed loader with explicit input budgets |
| Decider | Native typed API with independent questions |
| JevK5 | Native calibrated letter logits and candidate selection |
| Minojev | Native decision head and calibrated candidate scores |
| Luce | Standalone inference with recurrent backbone and typed criteria |
| Verdict2 | Requires a compatible `model.pt` checkpoint |
| Openvons | Requires a trained text checkpoint with `head.pt` |

Support for an adapter does not imply that a checkpoint is publicly available or
that every benchmark fits its context limit. Failed or partial runs remain visibly
incomplete. Do not interpret arbitrary upstream truncation or unavailable weights
as successful evaluation. Model and source licenses must be reviewed separately.
