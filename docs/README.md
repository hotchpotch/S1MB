# S1MB documentation

S1MB (System One Mosaic Benchmark) combines specialized Choice, Noul and Score
tasks. It does not establish unseen-task generalization or training-data non-overlap.

| Goal | Guide |
| --- | --- |
| Run a model, from smoke check to validated full run | [Evaluation](evaluation.md) |
| Put your model on the leaderboard via an HF Dataset PR | [Contributing results](contributing_results.md) |
| Display published results or compare a local run | [Viewer setup](viewer.md) |
| Implement and test a model integration | [Adapter development](adapters.md) |
| Work on the evaluator or viewer source | [Developer workflow](developer_workflow.md) and [contribution policy](../CONTRIBUTING.md) |
| Suggest a model for volunteers to evaluate | [Model evaluation requests](model_requests.md) |
| Understand metrics and aggregation | [Scoring specification](../evaluator/SCORING.md) |
| Deploy the maintained HF Space | [Space deployment](huggingface_space_deploy.md) |
| Prepare a source release | [Release checklist](../RELEASING.md) |

## Data and reference material

- [Evaluation dataset](https://huggingface.co/datasets/hotchpotch/s1mb-dataset):
  inputs and targets; consult the dataset card for its own access and license terms.
- [Results dataset](https://huggingface.co/datasets/hotchpotch/s1mb-result):
  compressed measurements and display metadata, submitted through Dataset PRs.
- [Evaluator reference](../evaluator/README.md) and
  [model runtime notes](../evaluator/OPEN_MODELS.md).
- [Viewer reference](../viewer/README.md) and
  [display cache architecture](../viewer/DISPLAY_DATA.md).

Source code, evaluation data and results have separate publication boundaries.
The code's MIT license does not grant rights to redistribute datasets or weights.
Do not add downloaded data, measurements or credentials to source PRs.
