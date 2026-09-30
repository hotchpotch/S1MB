# Developer workflow

Use this guide for source development and local checks. Contribution policy lives
in [Contributing](../CONTRIBUTING.md); deployment branch management and publishing
are consolidated in [Space deployment](huggingface_space_deploy.md).

## Set up a checkout

Read the repository's `AGENTS.md` and the guidance in the component you change.
Use Python 3.11 and uv for `evaluator/`, and Node.js 22.22.2/npm for `viewer/`.
Install dependencies in the checkout where you work:

```sh
# From evaluator/
uv sync --locked
```

```sh
# From viewer/
npm ci
```

Inspect `git status --short` and `git worktree list` before switching branches.
Worktrees share Git history, but dependencies, ignored data, and caches are local
to each checkout. Preserve the tracked `viewer/data` symlink to `../evaluator/data`.
Use explicit result-directory arguments to read measurements stored elsewhere.
Ordinary source contributions require no deployment access or deployment worktree.

## Choose the relevant workflow

| Change | Guide |
| --- | --- |
| Model integration | [Adapter contract and testing](adapters.md) |
| Evaluation behavior | [Evaluation runbook](evaluation.md) and [scoring specification](../evaluator/SCORING.md) |
| UI or local result comparison | [Viewer setup](viewer.md) |
| Display loading and recovery | [Prepared display data](../viewer/DISPLAY_DATA.md) |
| Result publication | [Dataset PR workflow](contributing_results.md) |
| Deployment branch or hosted viewer | [Space deployment](huggingface_space_deploy.md) |

To develop against an existing results folder, run from `viewer/`:

```sh
npm run prepare-display -- --results-dir /absolute/path/to/results
npm run dev
```

The wrapper prints a Tailscale IPv4 or localhost URL. See the [viewer guide](viewer.md)
for synchronization, combining sources, Docker, and restart behavior. The viewer
reads saved metrics; it does not evaluate models or download Hub updates itself.

## Keep generated artifacts local

Keep downloaded datasets, model weights, measurements, credentials, and generated
reports out of source commits. Standard ignored locations include:

| Content | Location |
| --- | --- |
| Evaluation inputs | `evaluator/data/datasets/` |
| Original local runs | `evaluator/data/results/<run-id>/` |
| Verified Hub snapshots | `evaluator/data/hub-results` and `evaluator/data/.hub-results*` |
| Recorded dataset revisions for validation | `evaluator/data/result-datasets/` |
| Audits and scratch reports | `evaluator/audits/`, `tmp/`, `output/`, `evaluator/output/` |
| Prepared display JSON and reports | `viewer/display/` |

Keep reports and auxiliary JSON outside result roots. Custom output destinations
must be outside the checkout or have their own ignore rules. Keep benchmark and
category definitions and shared synthetic fixtures tracked; do not ignore all of
`evaluator/data/` or all JSON/XZ files. Configuration names belong in
[`.env.sample`](../.env.sample); real credentials belong in ignored local files.

## Checks before sharing changes

From `viewer/`:

```sh
npm test
npm run typecheck
npm run build
```

The build includes Storybook, served under `/storybook/` by the production viewer.
Use synthetic stories to review visual changes at desktop and mobile sizes,
including comparison, model details and loading states.

From `evaluator/`:

```sh
uv sync --locked
uv run tox
```

Public CI must work without private data, credentials, model APIs or GPU access.
Use the dedicated dataset integration tests when authorized data is available;
do not make ordinary unit tests depend on it.

## Prepare a source contribution

Describe the problem, resulting behavior, and validation in the source PR. Keep
measurements in a separate results Dataset PR and link related code changes.
Before a source release, follow [RELEASING.md](../RELEASING.md), including review
of the release file list and Git history. Ignore rules do not remove previously
committed artifacts.
