# Requesting a model evaluation

If you would like to see a model on the S1MB leaderboard but cannot evaluate it
yourself, [open a model evaluation request](https://github.com/hotchpotch/S1MB/issues/new?template=model_evaluation.yml).
First search [existing requests](https://github.com/hotchpotch/S1MB/issues?q=is%3Aissue%20label%3Amodel-evaluation-request)
and add relevant details to an existing issue instead of opening a duplicate.

Requests are a community interest list. Contributors may evaluate a model when
they have spare time, suitable hardware, dataset access, and any required API
budget. **There is no guarantee of evaluation, a response, or a completion date.**
Please submit with that expectation; a request is not a reservation in a queue.

Include the model card or API documentation, exact model/version, access and
license requirements, hardware or API costs if known, and why comparison would
be useful. Mention an existing compatible adapter or the work needed to add one.
Do not post credentials, private download links, checkpoints or evaluation data.
Not every model supports the typed Choice, Noul and Score interfaces directly.

The issue form applies `model-evaluation-request`. Maintainers may close duplicate,
unavailable or unsupported requests with an explanation. If you want to volunteer,
comment with the model revision and planned scope before starting to avoid
unnecessary duplicate compute; that comment does not create a deadline.

You do not need an issue or an assignment to contribute your own measurements.
Follow [evaluation](evaluation.md), [adapter development](adapters.md) if needed,
and [adding your model to the leaderboard](contributing_results.md). Results go
through a PR to the [HF results dataset](https://huggingface.co/datasets/hotchpotch/s1mb-result/discussions),
not as GitHub issue attachments or source commits. Link the Dataset PR from the
request so others can follow progress. Publication still requires review and
synchronization; opening an issue does not add a leaderboard entry.
