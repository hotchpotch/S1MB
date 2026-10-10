# Hugging Face Space deployment

This is the maintainer runbook for deployment permissions, worktrees, branch
synchronization, prepared display data, and verification. For ordinary source changes,
use the [developer workflow](developer_workflow.md); for local hosting, use the
[viewer guide](viewer.md).

The deployment workflow bundles a public display JSON into the image. Keep raw
measurements, evaluation datasets and credentials out of its build inputs. Deployment preserves Space history,
checks the parent SHA, and pins the running image to an immutable digest.

## Deploying the Space

**Deployment currently requires a maintainer with the relevant permissions.**
Contributors without permission should submit their source changes for review and
ask an authorized maintainer to synchronize and deploy the deployment branch.
Do not copy another developer's credentials into a worktree.

The current arrangement is:

| Resource | Purpose | Visibility |
| --- | --- | --- |
| GitHub `hotchpotch/S1MB` | Source and deployment workflow | Private |
| GHCR `ghcr.io/hotchpotch/s1mb-leaderboard` | Docker image with public display JSON | Public |
| HF Space `hotchpotch/S1MB-leaderboard` | Hosted viewer | Public |
| HF Dataset `hotchpotch/s1mb-result` | Published measurements | Public |

A maintainer needs permission to push or trigger the repository's deployment
workflow. Its configured credentials must permit publishing the GHCR package and
updating the Space. Publishing prepared JSON additionally requires write access to the results Dataset.
The viewer itself needs no application HF token or mounted volumes.

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

Preserve the Space's existing visibility. Deployment supports public and private
Docker Spaces and never changes their visibility. Readiness checks reject a
visibility change during deployment and use anonymous HTTP for public Spaces.
Images contain a fixed display artifact; new measurements require preparation,
publication and a new deployment. There is no live Dataset mount or Bucket cache.

After a successful workflow, verify the leaderboard at
[the Space](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard), not just
that an image was published. If deployment fails, inspect the failed job before
retrying; a pushed branch alone is not evidence that the new app is running.

## Branches and worktrees

The deployment branch's exact name is **`hf-space-docker`**, not `hf_docker`.
`main` is the main development branch. Pushes to `hf-space-docker` trigger the
**Deploy HF Space** GitHub Actions workflow. A push to `main` alone does
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

## Automatically generated model links

On every normal deployment, the deploy script resolves the published results
Dataset at the measurement source revision recorded in the bundled JSON and reads only
its model-folder `metadata.json` files. It generates `models.py` at the Space
repository root alongside `README.md` and the digest-pinned `Dockerfile`.
The same list is written to the README YAML `models` metadata using the Hub card
serializer. Generated model references are not maintained by hand in source.

Like [HAKARI-Bench's model list](https://huggingface.co/spaces/hakari-bench/leaderboard/blob/main/models.py),
it contains a literal `MODEL_NAMES = [...]` list for Hub discovery. Hugging Face
documents automatic linking from Python files in
[Linking Models and Datasets](https://huggingface.co/docs/hub/spaces-overview#linking-models-and-datasets-on-the-hub).
The file is not imported by the viewer: it describes models represented in saved
results and does not load weights or run inference.

Generation follows these rules:

- Include model folders with at least one published benchmark `.json.xz` file.
  This is a reference list, not a declaration of complete evaluation coverage.
- Use `hf_url`, falling back to `url` when no explicit HF link exists. Skip
  API-only models and non-Hub links; do not infer a checkpoint from display names.
- Normalize checkpoint subfolders and revisions to `owner/model`. Deduplicate
  and sort IDs for deterministic output, without timestamps or changing result SHAs
  in the generated file.
- Read metadata at the same resolved SHA, validate its schema and folder identity,
  and reject missing, malformed, or oversized metadata. No results or weights are
  downloaded. Any fetch or validation failure stops deployment before Space writes.

Deployment compares all three generated files with the current Space revision
and commits only changed files, retaining the parent-SHA check. A model-list-only
change is included even when the image is unchanged. If every file is unchanged,
no Space commit is created. Removed model references disappear on the next deploy.

This runs on normal deployments, including manually dispatched deployments; there
is no scheduled Dataset watcher. Dataset synchronization alone does not update
`models.py`. Hub indexing and Space visibility determine whether visitors can see
the association; adding references does not make a private Space public.

To preview without modifying the Space, run from `evaluator/` with an actual image
digest and source SHA:

```sh
uv run python ../deploy/hf-space/deploy.py \
  --space hotchpotch/S1MB-leaderboard \
  --image ghcr.io/hotchpotch/s1mb-leaderboard@sha256:IMAGE_DIGEST \
  --source-sha SOURCE_COMMIT_SHA --results-revision DATASET_COMMIT_SHA --output ../tmp/space-preview
```

Preview still reads public results metadata from the Hub. `--results-repo` can
select another results repository for an explicit preview or deployment.

## Prepare and publish display data

Install viewer dependencies (`npm ci` in `viewer/`) and `xz` on the maintainer's
machine. From `evaluator/`, use the existing uv environment for the Hub SDK/Xet:

```sh
uv sync --locked
uv run python ../viewer/scripts/publish-display.py
```

This resolves the latest `hotchpotch/s1mb-result` commit, downloads only model
metadata and compressed measurements, validates their summaries and writes
`viewer/display/viewer-summary.json` plus a local `.report.json`. It does not need
private evaluation inputs. Publication validation of original predictions remains
the result submitter's responsibility. Review the reported source SHA and counts.
Preparation also fills missing parameter counts and recalculates legacy AP/TP
gaps of at most 1% using the static embedding-excluded definition in its temporary metadata copy
from supported public checkpoint headers. Review the local
`viewer-summary.json.parameters.report.json` for resolved checkpoint SHAs,
evaluation versus linked-checkpoint sources, and unresolved reasons. Original
result files and published model metadata remain unchanged. The checkpoint
counting rules and limitations are in the
[submission guide](contributing_results.md#model-metadata).

To stage the changed metadata files for a separate results Dataset PR, add
`--metadata-output ../tmp/parameter-metadata-submission`. This directory contains
only the changed model folders' `metadata.json` files; measurements are preserved.

To generate from the latest data and publish in one command:

```sh
uv run python ../viewer/scripts/publish-display.py --publish
```

A reduction against the previously published JSON stops generation and publication.
Review removed results, definitions, completion regressions and count/size changes.
Only after human confirmation, rerun with the exact candidate digest:

```sh
uv run python ../viewer/scripts/publish-display.py --publish \
  --approve-reduction REVIEWED_CANDIDATE_DIGEST
```

Do not automatically approve this warning in CI. A changed candidate requires a
new digest. Empty results always fail. First publication explicitly reports that
there is no previous artifact; source inventory is still checked. Upload adds only
the reserved root file `viewer-summary.json`. It preserves original measurements
and uses the resolved Dataset parent SHA to reject concurrent updates.

After publication, push `hf-space-docker` or dispatch the workflow on that branch.
CI resolves one Dataset commit for both Docker builds. `RESULTS_REVISION` is a
required build argument; the build downloads and validates just the public JSON
at that SHA. Repeating a deployment after publishing new data resolves a new SHA,
so Docker cannot reuse an older artifact layer by accident.

## Remove legacy mounts

After deploying an image containing prepared JSON, a Space administrator should
remove the old results Dataset and cache Bucket mounts. Inspect the volume list
first and preserve unrelated mounts. For a Space with only these two old mounts:

```python
from huggingface_hub import HfApi
api = HfApi()
api.delete_space_volumes("hotchpotch/S1MB-leaderboard")
```

The Bucket is no longer used; this detaches it without deleting its contents.
A new process needs only its image, so it no longer waits for remote cache reads.
Preserve the Space's existing public/private visibility.

Configuration names and credentials are documented in [`.env.sample`](../.env.sample).
See [display data architecture](../viewer/DISPLAY_DATA.md) for the format and
[local viewer instructions](viewer.md) for local conversion and Docker commands.
