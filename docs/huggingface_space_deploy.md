# Private Hugging Face Space deployment

See [Developer workflow](developer_workflow.md) for deployment permissions, the
`hf-space-docker` worktree, cross-merges with `main`, and routine deployment steps.

The private `hotchpotch/S1MB` repository builds a public, code-only GHCR image at
`ghcr.io/hotchpotch/s1mb-leaderboard`. The Docker Space
`hotchpotch/S1MB-leaderboard` stays private. Image visibility is independent of
repository and Space visibility. Do not include data or credentials in build inputs.

Pushes to `hf-space-docker` run `.github/workflows/deploy-space.yml`, validate the
code, publish a commit-tagged image, confirm anonymous image access, and deploy
its immutable digest. Deployment preserves Space history, checks the parent SHA,
and verifies the exact running Space revision and authenticated HTTP response.
The deployment token is a GitHub Actions secret, not a runtime dependency.

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
token is required. Defaults are `/mnt/results` and an hourly check. Override with
`S1MB_RESULTS_DIR` and `S1MB_RESULTS_CHECK_SECONDS` in Space Variables if needed.
The `--space` startup mode requires HF's `SPACE_ID` and binds port 7860 for the
managed proxy. Local Docker uses normal startup and safe host binding instead.

The first request restores a saved summary JSON from the cache Bucket and returns
it while the Node worker performs a full background rebuild. Only a missing or
invalid cache needs a synchronous initial load. During the same process lifetime,
changed source files alone are parsed/decompressed. Existing users receive cached
data while candidates are built and swapped. The source check interval is one hour.

The cache uses immutable, checksummed JSON generations, keeping the latest two
valid files per source/format namespace. Unchanged summaries produce no new file.
Read-back verification precedes cleanup; an interrupted generation is ignored,
with cleanup retried after a successful refresh. Mount-level asynchronous writes
can still lose their newest unflushed generation; retain the preceding generation.
The app does not force the Dataset mount to synchronize. Remote updates appear
only after the mount exposes them and a request triggers a source check.
`S1MB_RESULTS_CACHE_DIR` overrides the default `/mnt/cache`. Use one running viewer
writer per cache namespace. No runtime application token is needed for managed mounts.

GitHub repository variables: `S1MB_SPACE_REPO`, `S1MB_SPACE_IMAGE`.
GitHub secret: `S1MB_SPACE_DEPLOY_TOKEN` with write access to the private Space.
See `.env.sample` and [local viewer instructions](../viewer/README.md).

Official reference: [Space volumes](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces#mount-volumes-in-your-space).
