# Private Hugging Face Space deployment

The deployment branch is `hf-space-docker`. Its worktree is normally
`../S1MB-wt/hf-space-docker`. The Space name is `S1MB-leaderboard` and its
visibility must remain **private**. Results come from the public Dataset
[`hotchpotch/s1mb-result`](https://huggingface.co/datasets/hotchpotch/s1mb-result),
selected through runtime configuration. The private Space is
`hotchpotch/S1MB-leaderboard`; the public image is
`ghcr.io/hotchpotch/s1mb-leaderboard`. The linked GitHub repository
`hotchpotch/S1MB` remains private. These destinations are configured through
GitHub repository variables, not hardcoded in the workflow.

## Release flow

Make ordinary viewer changes on `main`, then merge `main` into the deployment
branch. Pushing that branch runs Python and viewer checks, builds the `space`
target of `viewer/Dockerfile`, and checks runtime imports without credentials,
private inputs, or a GPU. Manual workflow dispatch uses the same branch guard.

Once both deployment repository variables in [the environment sample](../.env.sample)
are configured, the workflow publishes a commit-tagged GHCR image and updates
the Space's `main` with a digest-pinned Dockerfile and the tracked Space card.
The Hub rebuilds that commit. The workflow waits for that exact Space revision
to reach `RUNNING`, then makes an authenticated request to the app. Failures are
reported by the workflow; it does not claim that uploading a commit means the
deployment is healthy. Deployments are serialized, and Hub commits use a parent
revision check. Existing history and unrelated Space files are preserved.

## One-time configuration

1. Choose the owner and create `OWNER/S1MB-leaderboard` as a **private Docker
   Space**. The deployment helper refuses a public Space or a different SDK.
2. Set the viewer results repository option to `hotchpotch/s1mb-result` in Space
   Variables. Public results do not require authentication; set a read-only token
   for private evaluation inputs in Space Secrets. See [the sample](../.env.sample) for exact
   setting names. The evaluation source remains `evaluator/dataset-source.json`.
3. Choose a GHCR image repository. HF must be able to pull this code-only image
   anonymously, independently of the Space's private visibility. Make the GHCR
   package public after its initial publication if necessary; the workflow stops
   before deploying if anonymous access fails. Rerun after changing visibility.
4. Configure the GitHub repository's deployment variables and deployment secret
   listed in [the sample](../.env.sample). The deployment token needs access to
   write the private Space and read its app; it is not passed into the image or
   viewer runtime. GHCR publication uses the workflow's GitHub token.
5. Push the deployment branch or manually dispatch its workflow. Inspect the
   Actions result and open the Space while signed in with an authorized account.

Unset deployment variables skip publication/deployment while still running
validation. No script creates a Space, changes visibility, or selects an owner
implicitly. Space secrets must be configured before the first deployment.

## Runtime data and networking

The image contains code, benchmark/category definitions, Node.js, `xz`, and a
locked minimal evaluator installation. It includes no downloaded datasets,
measurements, weights, local caches, or credentials. The existing `viewer/data`
relative symlink remains intact.

Before starting Next.js, the Space bootstrap materializes the configured current
evaluation release, then runs the existing results synchronization logic. This
validates saved results against their recorded evaluation revisions and prepares
those revisions as necessary. A failed preparation aborts startup. Python is a
startup process only; the running viewer is Node.js. Its data directory is fixed
to `/app/evaluator/data` in this image, with ephemeral storage. Cold starts repeat
preparation; persistent data volumes are not configured in this initial version.

The bootstrap also accepts `--results-repo hotchpotch/s1mb-result` and optional
`--results-revision main`. Arguments override the corresponding runtime settings
and are passed to both data preparation and the Node.js viewer. For an explicit
container command, use `/app/evaluator/.venv/bin/python /app/deploy/bootstrap.py`
followed by these arguments. Normally use Space Variables, so the deployment
preflight can check the repository configuration. No results repository default
is embedded in the image. Build arguments are not used for data or credentials.

The existing live Hub results mode checks on requests after its cache interval
(at least one hour). The JS cache initially downloads and validates results again;
it does not reuse the Python synchronization snapshot. New results using an
already prepared evaluation revision can refresh without restarting. Results
requiring a new evaluation revision retain the last validated snapshot and show
a refresh warning until the Space is restarted to prepare that data. Evaluation
data and definition changes also require restart. An open browser receives no
push updates.

Only the dedicated Space bootstrap binds `0.0.0.0:7860`, inside the managed
container. Local start behavior continues to bind Tailscale or localhost. Do not
use host networking with the Space entrypoint on a workstation. Private access is
enforced by Hugging Face; the app does not implement a separate login screen.

## Local validation and review

From `viewer/`, run `npm ci`, `npm test`, `npm run typecheck`, and `npm run build`.
From `evaluator/`, run `uv sync --locked`, `uv run tox`, and
`uv run pytest ../deploy/hf-space/test_deploy.py`. From the repository root:

```sh
docker build --target space -f viewer/Dockerfile -t s1mb-space:test .
```

To render reviewable Space files without Hub access, from `evaluator/`:

```sh
uv run python ../deploy/hf-space/deploy.py \
  --space OWNER/S1MB-leaderboard \
  --image ghcr.io/owner/image@sha256:REPLACE_WITH_64_HEX_DIGEST \
  --source-sha REPLACE_WITH_40_HEX_SOURCE_COMMIT \
  --output ../tmp/space-preview
```

Only `--deploy` enables Hub writes. It also waits for deployment verification.
For rollback, render and deploy a previously verified digest with its original
source commit, or revert the Space Dockerfile commit. Dataset rollback is a
separate operation; changing the image does not revert results or evaluation data.

References: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
and [Hub API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api).
