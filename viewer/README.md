# S1MB viewer

Node.js 22.22.2 and `xz` are required. Run npm commands in this directory.

```sh
npm ci
npm run dev -- --results-dir /absolute/path/to/results
# Production:
npm run build
npm start -- --results-dir /absolute/path/to/results
```

The default source is `data/hub-results` when present, otherwise `data/results`.
Repeat `--results-dir` to combine folders. Definitions default to `viewer/data`
(a symlink to `../evaluator/data`). No evaluation dataset is needed by the viewer.
See [`../.env.sample`](../.env.sample) for environment-based configuration.
The wrapper binds to Tailscale IPv4 when available, otherwise localhost.

## Remote leaderboard plus local evaluations

From `evaluator/`, install a verified remote snapshot into the Git-ignored
repository-root cache (the first synchronization may download recorded evaluation
revisions for validation):

```sh
uv run s1mb sync-results --repo-id hotchpotch/s1mb-result \
  --output ../cache/hf-s1mb-result
```

Then, from `viewer/`, combine that snapshot with local runs:

```sh
npm run dev -- --results-dir ../cache/hf-s1mb-result \
  --results-dir ../evaluator/data/results
# Or, after npm run build:
npm start -- --results-dir ../cache/hf-s1mb-result \
  --results-dir ../evaluator/data/results
```

Use a specific `../evaluator/data/results/<run-id>` directory to show only one
local run alongside the remote leaderboard. Both directories must exist.
Repeat synchronization to update remote results, then restart the viewer after
remote or local results change. Synchronization leaves the previous verified
snapshot installed on failure. The viewer does not pull from the Hub itself.
A fully downloaded Git/Xet checkout with the same published layout can also be
passed as a results directory; pointer files are not result data. Prefer the
validated synchronization command above for atomic updates.

Published model folders retain their model IDs and display metadata; local runs
retain their run IDs, so the same model can appear as separate remote and local
rows. Sources are combined without overriding conflicting measurements. Identical
row/benchmark results are deduplicated; conflicting identities or results reject
the refresh and preserve the last good viewer snapshot.

Models completing just the active Generalization benchmarks (currently six:
Diverse and Contextual × Noul, Choice and Score) appear on the overall leaderboard
with General scores. Missing full-category aggregates remain unavailable and
coverage is shown explicitly. Select **Generalization tasks only** to rank within
that subset. Evaluation selection from `evaluator/` is:

```sh
uv run s1mb run --adapter YOUR_ADAPTER --model YOUR_MODEL \
  --generalization-only --run-id YOUR_FRESH_RUN_ID
```

This option intersects with `--category`, `--task`, and repeated `--benchmark`
filters. It does not establish unseen-task generalization or training-data non-overlap.

## Filesystem cache

See [Display data and cache lifecycle](DISPLAY_DATA.md) for the source-to-display
flow, background updates, generation retention, recovery and implementation map.

On first access after startup, the worker restores the newest valid, checksummed
summary JSON from disk. It returns this snapshot immediately while rebuilding
from all original files in the background. Only an installation without a valid
cache waits for the first full load. No file index is persisted across restarts.
Subsequent checks reuse the worker's per-file summaries and decode only changes.
A second metadata scan rejects changes occurring during loading. Candidates
replace the memory cache atomically; errors retain the last good snapshot.

The default cache directory is `viewer/.cache/results` (Git-ignored). Override it
with `--cache-dir /path/to/cache`. Each source configuration
and cache format has its own namespace; use one viewer writer per namespace.
The JSON contains display summaries only, never raw inputs or predictions.
New immutable generations are written only when summary content changes, closed,
read back and verified, then older and invalid generations are removed. Keep the
latest two valid generations, with a temporary third while saving. Interrupted
writes are ignored on restoration and cleaned after the next successful source
refresh/save. Unknown formats or an entirely corrupt cache trigger a rebuild.
A persistence failure does not discard fresh measurements; the next source check
retries the pending save. The server logs persistence failures.

On managed mounts, remote durability follows the mount's flush semantics; a
local read-back is not proof that an asynchronous remote upload has completed.
The preceding good generation remains available for recovery. Local files use
`fsync` before close where supported. Cache directories are disposable and should
not be shared by independently running viewer deployments.

Local startup checks after every request by default. Set `--check-seconds 3600`
to check at most hourly. Concurrent requests
share one refresh. There is no polling when nobody accesses the viewer. A check
starts on the first request after the interval; that request receives old data,
and a subsequent request sees successfully refreshed results.

Saved baseline metrics drive adjusted scores. The viewer validates summaries,
counts and metadata; it does not re-evaluate predictions against private targets.
Use the evaluator's publication validation before uploading results. Current
benchmark/category definitions control membership. Raw predictions, provenance,
full model settings and evaluation inputs are not sent to the browser.

## Local Docker

Build from the repository root, then use a read-only bind mount and host networking
(on Linux) so the same safe address selection works:

```sh
docker build -f viewer/Dockerfile -t s1mb-viewer .
docker run --rm --network host \
  --mount type=bind,src=/absolute/path/to/results,dst=/mnt/results,readonly \
  --mount type=volume,src=s1mb-viewer-cache,dst=/mnt/cache \
  s1mb-viewer npm start -- --results-dir /mnt/results --cache-dir /mnt/cache --port 3000
```

Both Docker targets contain Node.js and `xz`, with no Python or Arrow dependency.
Only source, UI assets and benchmark/category definitions are in the image.

## Hugging Face Spaces

See [deployment](../docs/huggingface_space_deploy.md). Attach the public results
Dataset as a read-only volume at `/mnt/results`, and a private writable cache
Bucket at `/mnt/cache`. The managed mount handles Hub/Xet
reads and remote updates. The app only reads this filesystem; its hourly scan is
separate from the mount's own synchronization interval. Local use can bind a
regular results directory or an externally managed `hf-mount` directory. A regular
local directory does not download Hub updates by itself.

## Checks

```sh
npm test
npm run typecheck
npm run build
```

Tests use synthetic files without private datasets or credentials. Storybook is
built under `/storybook/`; its measurements are synthetic.
