---
title: S1MB Leaderboard
emoji: 🧩
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
fullWidth: true
license: mit
---

# S1MB Leaderboard

System One Mosaic Benchmark combines specialized Choice, Noul, and Score tasks.
It does not establish unseen-task generalization or training-data non-overlap.

This Space runs an immutable viewer image built from the source
repository's `hf-space-docker` branch. Results are read from a managed, read-only Dataset volume; the browser receives
compact summaries only. Evaluation inputs are not needed by this viewer. Dataset rights are
separate from the source code's MIT license.

## Models represented in the leaderboard

[models.py](models.py) lists Hugging Face model repositories with published S1MB
results. These literal references support the Hub's automatic model/Space
linking. The README `models` metadata contains the same list. Both are generated from published model metadata on each deployment and
updated when their contents change. The Python file is not an inference entry point. The viewer displays saved measurements without loading model weights.
