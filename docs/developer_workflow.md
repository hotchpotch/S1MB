# Developer workflow

This guide covers source development, local results, and deployment of the
leaderboard. Read the repository's `AGENTS.md` and the guidance in the component
you are changing. See [code contributions](../CONTRIBUTING.md) for contribution
policy and [evaluation](evaluation.md) for running models.

## Component development

For a new model integration, follow [Developing a model adapter](adapters.md).
For local UI setup and result previews, follow [Viewing the leaderboard](viewer.md).
The [documentation index](README.md) links evaluation, result submission, scoring,
and community request guidance. Ordinary source contributions do not require
Space deployment access or a deployment worktree.

## Branches and worktrees

The deployment branch's exact name is **`hf-space-docker`**, not `hf_docker`.
`main` is the main development branch. Pushes to `hf-space-docker` trigger the
**Deploy private HF Space** GitHub Actions workflow. A push to `main` alone does
not deploy the Space.

Developers may keep the deployment branch in a separate worktree, for example
`../S1MB-wt/hf-space-docker`. This is a workspace convention, not a required path.
Always inspect the actual checkout layout before switching branches or editing:

```sh
git worktree list
git status --short
git branch -vv
```

If `hf-space-docker` is already checked out in another worktree, do deployment
branch work **in that worktree**. Do not create a duplicate checkout or force a
branch switch in the main checkout. For a new setup, after `git fetch origin`,
use one of the following from the main repository:

```sh
# The local branch already exists and is not checked out elsewhere:
git worktree add ../S1MB-wt/hf-space-docker hf-space-docker

# Or create a local tracking branch when only the remote branch exists:
git worktree add -b hf-space-docker ../S1MB-wt/hf-space-docker origin/hf-space-docker
```

Worktrees share Git history and branches, but ignored data, dependencies and
caches are local to each checkout. Install dependencies in the worktree where
you work. Keep `viewer/data` as the tracked relative symlink to
`../evaluator/data`; use explicit result-directory arguments to read results
stored elsewhere.

## Keep main and deployment synchronized

Normally keep `main` and `hf-space-docker` current through **cross-merges**:
merge current `main` into the deployment branch, validate the combined result,
then bring that result back into `main`. Preserve both branches' changes; avoid
rebasing or force-pushing shared deployment history.

The example below assumes the main checkout is `S1MB` and the deployment worktree
is its sibling `S1MB-wt/hf-space-docker`. Adjust paths to `git worktree list`.
Start with clean worktrees; coordinate with anyone using them and preserve their
uncommitted work.

```sh
# In the main checkout:
git fetch origin
git merge --ff-only origin/main

# In the deployment worktree:
cd ../S1MB-wt/hf-space-docker
git merge --ff-only origin/hf-space-docker
git merge main
# Resolve conflicts, inspect the result, and run the checks below.

# Back in the main checkout, after validation:
cd ../../S1MB
git merge --ff-only hf-space-docker

# Authorized maintainers publish both branch updates together:
git push --atomic origin main hf-space-docker
```

A fast-forward is sufficient when only one branch has advanced; if both already
point to the same commit, no merge commit is needed. If a fast-forward fails,
inspect the new history and reconcile it rather than resetting someone else's
changes. If `main` advances during validation, merge it again in the deployment
worktree and revalidate the combined result before publishing.

## Local setup and startup

Use Node.js 22.22.2, npm, and `xz` (`xz-utils` on Debian/Ubuntu). Evaluator commands
use Python 3.11 and `uv`, run from `evaluator/`. Run npm commands from `viewer/`.

For an existing results directory, no evaluation dataset or Python runtime is
needed by the viewer:

```sh
cd viewer
npm ci
npm run dev -- --results-dir /absolute/path/to/results

# Production mode:
npm run build
npm start -- --results-dir /absolute/path/to/results
```

Without an explicit result directory, startup selects `data/hub-results` if it
exists, otherwise `data/results`. Repeat `--results-dir` to combine sources.
The directories must exist. The start wrapper binds to the machine's Tailscale
IPv4 address when available, otherwise localhost; use its printed URL. Keep local
services off public interfaces. The managed Space startup mode is not for local
use.

For local Docker with a read-only result mount and a persistent cache volume,
use the [viewer Docker instructions](../viewer/README.md#local-docker).
Environment-based configuration is listed in [`.env.sample`](../.env.sample).
Never put credentials in tracked files or Docker build inputs.

## Hugging Face results Dataset

Published measurements live in the public
[`hotchpotch/s1mb-result`](https://huggingface.co/datasets/hotchpotch/s1mb-result)
Dataset, separately from the source repository and the evaluation dataset.
Each published model folder contains:

```text
<org-or-user>__<model-id>/
  metadata.json
  <benchmark-id>.json.xz
```

A folder represents one leaderboard row. `metadata.json` supplies display
metadata; compressed benchmark files contain the measurements. Different files
may record different original run IDs or evaluation revisions. Preserve their
provenance and validate each against its recorded dataset revision.

To install a verified snapshot locally, run from `evaluator/`:

```sh
uv sync --locked
uv run s1mb sync-results --repo-id hotchpotch/s1mb-result \
  --output ../cache/hf-s1mb-result
```

The results Dataset is public, but full publication validation may require access
to the recorded evaluation dataset revisions. Authenticate with an authorized
account if required; source-repository access does not grant dataset access.
A failed synchronization leaves the previous verified snapshot installed.

Then, from `viewer/`, display remote results together with local evaluations:

```sh
npm run dev -- --results-dir ../cache/hf-s1mb-result \
  --results-dir ../evaluator/data/results
```

Local runs retain their run IDs. Published rows retain their published identity.
Identical row/benchmark measurements can be deduplicated; conflicting results
reject the refresh rather than silently overriding another source.

Repeat synchronization to obtain new Hub results. The viewer reads files; it
does not download Hub updates itself. Changes within configured sources are
checked in the background on eligible requests. Restart when changing source
paths, or when a newly created default `data/hub-results` should replace the
previously selected local source.

Use `export-results`, `validate-results`, and a Dataset PR to publish measurements;
follow [the result submission guide](contributing_results.md) for exact commands
and review requirements. Keep raw runs, downloaded data, validation caches and
generated reports out of the source repository. Do not edit measurements merely
to change their display metadata. A result publication updates the Dataset;
a code change updates the source repository and requires a new app deployment.

## Display cache and updates

Local startup and Docker use the same loader as the Space. A separate Node child
process validates summaries and decompresses only changed files during its
lifetime. Local checks default to every eligible request; Space checks default
to at most hourly. Checks are request-driven, with no idle polling. Cached requests
continue receiving the previous snapshot while a refresh runs.

The display-only JSON cache restores a valid snapshot after restart, then starts
a full source rebuild in the background. It retains two verified generations per
source/format namespace and writes no new generation when summaries are unchanged.
A failed refresh preserves the last good snapshot. The loading screen covers the
initial data wait once Next.js is serving requests; it cannot replace HF's screen
before the container starts.

See [display data and cache lifecycle](../viewer/DISPLAY_DATA.md) for storage,
recovery, limitations and implementation references.

## Deploying the private Space

**Deployment currently requires a maintainer with the relevant permissions.**
Contributors without permission should submit their source changes for review and
ask an authorized maintainer to synchronize and deploy the deployment branch.
Do not copy another developer's credentials into a worktree.

The current arrangement is:

| Resource | Purpose | Visibility |
| --- | --- | --- |
| GitHub `hotchpotch/S1MB` | Source and deployment workflow | Private |
| GHCR `ghcr.io/hotchpotch/s1mb-leaderboard` | Code-only Docker image | Public |
| HF Space `hotchpotch/S1MB-leaderboard` | Hosted viewer | Private |
| HF Dataset `hotchpotch/s1mb-result` | Published measurements | Public |
| HF Bucket `hotchpotch/s1mb-leaderboard-cache` | Persistent display JSON | Private |

A maintainer needs permission to push or trigger the repository's deployment
workflow. Its configured credentials must permit publishing the GHCR package and
updating the private Space. Initial setup or changes to secrets, Space volumes,
and Bucket permissions require corresponding administrative access. The viewer
itself does not need an application HF token for managed mounts.

After cross-merging and passing checks, push `hf-space-docker`. The workflow
validates the code, builds and publishes the image, checks anonymous image access,
and deploys an immutable image digest. It then checks the exact running Space
revision and an authenticated HTTP response. Monitor it from an authorized shell:

```sh
gh run list --workflow deploy-space.yml --branch hf-space-docker
gh run watch RUN_ID --exit-status
```

To redeploy the current deployment branch without a new source commit:

```sh
gh workflow run deploy-space.yml --ref hf-space-docker
```

Keep the Space private. The results Dataset is mounted read-only at `/mnt/results`
and the cache Bucket read-write at `/mnt/cache`. Hub/Xet synchronization is handled
by the managed mount, independently of the viewer's hourly check. Dataset changes
become visible after both mount synchronization and an eligible request; an exact
one-hour publication-to-display deadline is not guaranteed.

The authoritative setup and volume configuration are in
[Private Hugging Face Space deployment](huggingface_space_deploy.md). After a
successful workflow, verify the authenticated leaderboard at
[the Space](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard), not just
that an image was published. If deployment fails, inspect the failed job before
retrying; a pushed branch alone is not evidence that the new app is running.

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
