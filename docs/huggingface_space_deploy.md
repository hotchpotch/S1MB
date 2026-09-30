# Private Hugging Face Space deployment

This is the maintainer runbook for deployment permissions, worktrees, branch
synchronization, managed volumes, and verification. For ordinary source changes,
use the [developer workflow](developer_workflow.md); for local hosting, use the
[viewer guide](viewer.md).

The deployment workflow builds a code-only image. Keep datasets, measurements,
and credentials out of its build inputs. Deployment preserves Space history,
checks the parent SHA, and pins the running image to an immutable digest.

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

After a successful workflow, verify the authenticated leaderboard at
[the Space](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard), not just
that an image was published. If deployment fails, inspect the failed job before
retrying; a pushed branch alone is not evidence that the new app is running.

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
# Resolve conflicts, inspect the result, and run the checks in docs/developer_workflow.md.

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

## Mounted results

Configure a read-only Dataset volume using the Hub SDK from an administrative
machine (this replaces the volume list, so inspect existing volumes first):

```python
from huggingface_hub import HfApi, Volume
api = HfApi()
api.create_bucket("hotchpotch/s1mb-leaderboard-cache", private=True, exist_ok=True)
api.set_space_volumes("hotchpotch/S1MB-leaderboard", [
    Volume(type="dataset", source="hotchpotch/s1mb-result",
           mount_path="/mnt/results", read_only=True),
    Volume(type="bucket", source="hotchpotch/s1mb-leaderboard-cache",
           mount_path="/mnt/cache", read_only=False),
])
```

The managed filesystem supplies Xet-backed results. The runtime is Node.js plus
`xz`; no Python bootstrap, private evaluation dataset, Arrow, or application HF
token is required. Defaults are `/mnt/results` and an hourly check. Space variable
names and configuration are listed in [`.env.sample`](../.env.sample).
The dedicated `--space` startup mode requires the managed Space environment and
binds to `0.0.0.0:7860` for the managed proxy. Local Docker uses normal startup
and safe host binding instead.

When a valid display snapshot exists in the cache Bucket, the first request
restores it, then starts a full source rebuild in the background. Without a valid
cache, the first request waits for a full source load. Subsequent checks are
incremental within the worker's lifetime. Refresh failures preserve the last good
snapshot; two verified generations support recovery. Use one viewer writer per
cache namespace. See [cache architecture](../viewer/DISPLAY_DATA.md) for persistence,
mount durability, and recovery details.

The app does not force the Dataset mount to synchronize. Remote updates appear
only after the mount exposes them and a request triggers a source check. An exact
one-hour publication-to-display deadline is not guaranteed.

Configure workflow variables and the deployment secret using the names in
[`.env.sample`](../.env.sample). The deployment credential needs write access to
the private Space; it is not passed into the viewer runtime.

Official reference: [Space volumes](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces#mount-volumes-in-your-space).
