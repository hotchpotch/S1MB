# Private Hugging Face Space deployment

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
api.set_space_volumes("hotchpotch/S1MB-leaderboard", [
    Volume(type="dataset", source="hotchpotch/s1mb-result",
           mount_path="/mnt/results", read_only=True),
])
```

The managed filesystem supplies Xet-backed results. The runtime is Node.js plus
`xz`; no Python bootstrap, private evaluation dataset, Arrow, or application HF
token is required. Defaults are `/mnt/results` and an hourly check. Override with
`S1MB_RESULTS_DIR` and `S1MB_RESULTS_CHECK_SECONDS` in Space Variables if needed.
The `--space` startup mode requires HF's `SPACE_ID` and binds port 7860 for the
managed proxy. Local Docker uses normal startup and safe host binding instead.

The first request loads summaries. Subsequent requests immediately return the
cached snapshot and, after the interval, trigger a separate Node process to scan
filesystem metadata. Only changed files are parsed/decompressed. Candidate
snapshots replace the old cache atomically; errors retain the previous snapshot.
The application does not force the mount to synchronize: a remote update appears
only after the mount exposes it, a request triggers an eligible scan, and that
scan completes. No users means no application refresh work.

GitHub repository variables: `S1MB_SPACE_REPO`, `S1MB_SPACE_IMAGE`.
GitHub secret: `S1MB_SPACE_DEPLOY_TOKEN` with write access to the private Space.
See `.env.sample` and [local viewer instructions](../viewer/README.md).

Official reference: [Space volumes](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces#mount-volumes-in-your-space).
