# Display data and cache lifecycle

The viewer uses the same Node.js filesystem loader in local development, local
Docker and Hugging Face Spaces. It reads saved result metrics from JSON/XZ files
and produces a compact display snapshot. Python, Arrow and evaluation inputs are
not required. `xz` is used to decompress published result files.

## Sources and responsibilities

| Layer | Responsibility |
| --- | --- |
| Results directory | Original local JSON results or published model folders containing `metadata.json` and benchmark `.json.xz` files |
| Benchmark/category definitions | Current leaderboard membership and grouping |
| Node child process | Filesystem scans, decompression, summary validation, persistent cache reads and writes |
| Web process | Keep the current snapshot in memory and render requests |
| Persistent JSON cache | Restore a previously validated display snapshot after a restart |

Published folder identity and `metadata.json` provide leaderboard row identity
and display metadata. Local runs retain their run identity. See the
[publication layout and validation guide](../docs/contributing_results.md).

The snapshot includes definitions, display metadata, saved metrics, counts,
completion status and baseline eligibility. Raw inputs, per-example predictions,
full model settings and credentials are not part of the display cache. The viewer
checks metric ranges, counts, membership and saved baseline adjustment; it does
not validate predictions against evaluation targets. Publication validation
remains the evaluator's responsibility.

Task scores and Task Avg follow [the scoring specification](../evaluator/SCORING.md):
adjust and clip each benchmark score, average within each task, then weight the
three tasks equally and multiply by 100. Complete coverage is required. Detail
views retain the primary metrics and their original directions.

## First access after startup

1. The worker looks for the newest valid display JSON in the configured cache
   namespace. It checks the format, source identity, checksum and summary schema.
2. When a valid generation exists, the web process serves that snapshot without
   waiting for a scan of the original result tree.
3. A full source rebuild starts in the background immediately, regardless of the
   normal refresh interval. The restored snapshot stays available meanwhile.
4. After successful validation, the new snapshot replaces the in-memory snapshot.

If the newest generation is corrupt or incomplete, restoration tries the previous
generation. If no valid generation exists, the first request waits for a full
source load. A source failure at this point cannot be served from a cache and is
reported as an error.

Restoration still requires reading and validating the display JSON; it is not a
zero-I/O operation. A remote cache mount can add latency. No per-file index or
inode information is persisted, so every process restart performs a full rebuild
after restoration. This deliberately avoids reusing filesystem identities across
container and mount lifetimes.

## Updates while running

Checks are request-driven, not a periodic polling job. When a request arrives
after the check interval, it receives the current snapshot and starts a check in
the child process. Concurrent requests share one check. The next interval starts
when that check finishes, including when it fails. With no requests, no new
checks start.

The worker scans file metadata, including size, timestamps and inode. If nothing
changed, it does not read or decompress result contents. Otherwise it reuses
unchanged per-file summaries and parses only added or changed files. Deletions
are reflected in the candidate snapshot. Metadata-only changes can reuse decoded
results. Benchmark and category definitions participate in the same check.

The worker assembles and validates a complete candidate, then scans metadata
again to detect source changes during loading. Only a successful candidate
replaces the active snapshot. A source error leaves the previous snapshot intact
and sets the refresh-failure indication. A later eligible request retries.

The browser does not receive a push update: a later request or page reload sees
the refreshed data. The displayed last-check time advances after a successful
check even if summary content did not change.

Incremental processing applies within the running worker. The persisted JSON is
a complete snapshot, not a patch log. This keeps recovery independent of earlier
generations and avoids a growing chain of deltas.

## Persistence and bounded generations

Each cache namespace is derived from the format version, configured absolute
definition directory and ordered result-directory paths. It does not identify a
Hub commit. Use one viewer writer per namespace.

```text
<cache directory>/
  v1-<source hash>/
    <timestamp>-<uuid>.json
    <timestamp>-<uuid>.json
```

Each generation contains `version`, `source`, `savedAt`, a SHA-256 `digest` and
the display `snapshot`. Runtime refresh status is not persisted. The format
version must change when snapshot semantics or representation become incompatible.

When normalized summary content changes, the worker writes a new immutable file,
flushes it where supported, closes it, and reads it back for verification. Only
then does it remove older generations. It retains the latest two valid generations
per namespace; a third can exist temporarily during a save. Unchanged content
does not create another generation, even after a restart and full rebuild.

Interrupted or invalid generations are skipped during restoration and removed
after a successful save or unchanged-content verification. Interrupted cleanup
can temporarily leave more than two files. Cleanup only touches recognized
generation filenames in the active namespace. Old namespaces from different
source configurations or format versions are not automatically removed.

A persistence failure does not discard a newly loaded valid snapshot. The worker
keeps a pending save and retries it on the next source check, including when the
source files are unchanged. Source results remain authoritative; the display
cache is disposable.

For managed mounts, local read-back verification does not prove that an
asynchronous remote upload has completed. Remote durability follows the mount's
flush behavior. Keeping the previous valid generation provides a recovery option,
but this is not a distributed transaction or a multi-writer cache.

## Configuration by environment

| Setting | Local startup / local Docker | Managed Space |
| --- | --- | --- |
| Results directory | `data/hub-results` if present, otherwise `data/results` | `/mnt/results` |
| Cache directory | `viewer/.cache/results` | `/mnt/cache` |
| Check interval | 0 seconds: check on each eligible request | 3,600 seconds |

Use `--results-dir` (repeatable), `--cache-dir` and `--check-seconds` with the start
wrapper to override these defaults. Environment alternatives are listed in
[`.env.sample`](../.env.sample).
The definition directory defaults to `viewer/data`, the symlink to
`../evaluator/data`. Source selection occurs at startup; changing
the selected paths requires a restart.

For normal local use, generate the display cache automatically on first access.
An existing cache is restored on subsequent starts. A regular local results
directory does not download Hub updates: synchronize it externally or use a
managed mount. The viewer notices filesystem changes on the next eligible access.

For local Docker, mount results read-only and use a named volume for the cache,
passing `--cache-dir` for its container path. Without a persistent cache volume,
recreating the container can require a full initial load. See the runnable
[Docker example](../docs/viewer.md#local-docker).

The private `hotchpotch/S1MB-leaderboard` Space uses the public
`hotchpotch/s1mb-result` Dataset mounted read-only at `/mnt/results`, and the
private `hotchpotch/s1mb-leaderboard-cache` Bucket mounted at `/mnt/cache`.
The managed Dataset mount handles Hub/Xet reads. The viewer itself reads only the
filesystem and has no Hub download loop. Its hourly check is separate from the
mount's synchronization schedule, so a Hub publication is not guaranteed to
appear within exactly one hour. The metadata scans do not pin an entire mounted
tree to a single Hub revision. See [Space deployment](../docs/huggingface_space_deploy.md).

## Operations and implementation references

Useful server log messages include `Restored results cache`, `Saved results cache`,
`Ignoring invalid results cache generation`, `Results refresh failed` and
`Results cache persistence failed; will retry on the next check.` If restoration
fails repeatedly, check directory permissions and available storage. Rebuilding
without the cache is supported; do not remove the authoritative result files.

The implementation was verified with synthetic tests and real-result checks for
normal startup, Docker container recreation with a cache volume, and private
Space restart with the Bucket cache. These checks confirmed restoration followed
by background rebuilding, and no additional generation for unchanged summaries.

| Implementation | Purpose |
| --- | --- |
| [Start wrapper](scripts/start.ts) | Source selection, paths and interval defaults |
| [Request cache](src/lib/results.ts) | Immediate cached responses, refresh coalescing and worker lifecycle |
| [Worker](scripts/results-worker.ts) | Separate process, restoration and pending-save retries |
| [Filesystem loader](src/lib/result-loader.ts) | Incremental scans, validation and complete candidates |
| [Result decoding](src/lib/result-files.ts) | Bounded JSON/XZ loading |
| [Snapshot store](src/lib/snapshot-store.ts) | Versioned JSON, checksums, recovery and retention |
| [Tests](src/lib/results.test.ts) | Synthetic loader and cache regression coverage |

## Loading screen

The route's `app/loading.tsx` streams a lightweight loading screen while the first
snapshot is restored or built. It automatically gives way to the dashboard when
data is ready. The screen has no estimated percentage, respects reduced-motion
preferences and exposes a status message to assistive technology. Normal background
refreshes continue serving the existing dashboard without showing this fallback.

This applies once Next.js can accept requests. While the Space container itself
is starting or sleeping, availability and the startup screen are controlled by
Hugging Face. Proxies may also affect when streamed HTML becomes visible.
