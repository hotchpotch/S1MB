---
title: S1MB Leaderboard
emoji: 📊
short_description: Compare System One models across 100+ specialized benchmarks.
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
fullWidth: true
license: mit
---

# S1MB Leaderboard

### Comparing System One Decision Models across 100+ benchmarks

**S1MB (System One Mosaic Benchmark)** evaluates **System One Decision Models**:
models that turn text and instructions into structured decisions. It brings three
decision types into a shared evaluation workflow:

- **Choice:** select an option from a set of alternatives.
- **Noul:** make a yes/no judgment using the supplied definition.
- **Score:** assign a numeric score using the supplied criteria.

The mosaic combines tasks from public NLP datasets with synthetic tasks that vary
instructions, contexts, and decision criteria. Use this leaderboard to compare
models across tasks, explore individual benchmark results, and see where a model
performs well or struggles. Missing or failed measurements remain visibly incomplete.

S1MB provides a practical reference point for comparison within these tasks.
Its scores do not establish unseen-task generalization or training-data non-overlap.

[Read the benchmark introduction](https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark/)
· [Explore the evaluation dataset](https://huggingface.co/datasets/hotchpotch/s1mb-dataset)
· [Browse published results](https://huggingface.co/datasets/hotchpotch/s1mb-result)

## About this Space

This is an interactive viewer of saved benchmark measurements; it does not load
model weights or run inference. The Docker image includes compact display data
prepared from the public results Dataset. New results appear after the display
data is regenerated and the Space is redeployed.

The viewer's source code is MIT-licensed. Evaluation datasets and model checkpoints
retain their own licenses and redistribution terms.

## Models represented in the leaderboard

[models.py](models.py) lists Hugging Face model repositories with published S1MB
results. These literal references support the Hub's automatic model/Space
linking. The README `models` metadata contains the same list. Both are generated from published model metadata on each deployment and
updated when their contents change. The Python file is not an inference entry point. The viewer displays saved measurements without loading model weights.
