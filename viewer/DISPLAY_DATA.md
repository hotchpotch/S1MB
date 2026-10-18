# Prepared display data

The viewer reads one immutable `viewer-summary.json` from local disk. It never
scans result folders, decompresses XZ, contacts the Hub, starts a refresh worker,
or reads a persistent Bucket at runtime. Build-time preparation validates the JSON and generates plain, gzip and Brotli responses
with a content-addressed manifest. Each server process caches those prepared bytes
once, without parsing the artifact, calculating hashes or compressing responses.
Updating the file requires a restart; updating a Space requires a new image.

## Publication lifecycle

1. Publish original measurements through the normal results Dataset PR workflow.
2. On a maintainer's machine, `scripts/publish-display.py` resolves the latest
   `hotchpotch/s1mb-result` commit once and downloads model metadata and XZ results
   at that exact SHA through the Hub SDK/Xet. Public reads need no credentials.
3. The Node converter validates saved scores, counts, metadata and current
   benchmark/category membership. It does not re-evaluate private targets;
   publication validation remains the evaluator's responsibility.
4. Compare the candidate against the previously published display JSON. Stop on
   removed measurements or definitions, completion regressions, fewer successful
   decisions, or a greater than 20% reduction in summary content bytes. The report
   includes previous/current model, benchmark, result and complete-result counts.
5. After review, publish only `viewer-summary.json` to the Dataset root. Upload
   requires Dataset write access and checks the original parent SHA. If another
   commit appeared, regenerate from the new latest revision; do not force a write.
6. Build the Docker image with the exact Dataset SHA containing the artifact.
   Both CI image builds use the same resolved revision. The image fetches only
   the public JSON, validates it and prepares delivery bytes before inclusion.
   The source JSON is removed from the runtime image. Deploy that digest.

For commands, see [Space deployment](../docs/huggingface_space_deploy.md).
`viewer-summary.json` is a reserved root-level generated artifact, not a model
folder or measurement. Results synchronization selects model files only. The
converter explicitly ignores the artifact when reading a full Dataset checkout.
Raw measurements and their history remain unchanged.

## Format and validation

Format version 1 contains a generation timestamp, source Dataset ID and exact
measurement commit SHA, SHA-256 digest, and a compact payload. The payload stores
distinct model configurations and benchmark descriptions once and refers to them
by index from results. Multiple configurations may share a published model-folder
identity, so measurements from different runs or languages retain their display
settings while contributing to the same leaderboard row. Every model entry must
be referenced, and duplicate identity/configuration pairs are rejected.
Recorded benchmark descriptions retain revision-specific counts; current
definitions separately control display membership. Raw inputs, predictions,
credentials and full model settings are excluded.

Schema, references, checksum and result consistency are validated before serving.
The maximum artifact size is 32 MiB. Files are written through an exclusive
same-directory temporary file, flushed and atomically renamed. An invalid
candidate or unapproved reduction cannot replace the previous output. There are
no cache generations or patch chains. Dataset Git history provides earlier
published versions, and earlier image digests provide deployment rollback points.

An approval is the candidate's content digest, not an unconditional bypass flag.
A changed candidate requires a new review. Empty output is always rejected.
On the first publication there is no previous artifact to compare; the report
explicitly says so. Original source inventory is checked before publication.
Reports remain local and are never uploaded alongside measurements.

## Local use and loading

Use `npm run prepare-display` to convert local result folders into the same format,
or `npm run fetch-display` to download an already published artifact at a pinned
SHA. Run `npm run prepare-delivery -- [DISPLAY_FILE]` explicitly before startup.
Start with `--display-file` (default `viewer/display/viewer-summary.json`).
See [local viewer instructions](../docs/viewer.md).

The initial page is a statically rendered loading shell. It contains no result
snapshot, so its response size is independent of the number of measurements.
The shell preloads `/api/display/` from the same viewer, so the display transfer
can overlap with JavaScript loading. After hydration, the browser fetch uses
matching credentials so supporting browsers can reuse the preload. The
server reads the prepared plain/gzip/Brotli bytes once and negotiates compression with
`Accept-Encoding`, preferring Brotli when its accepted quality is at least gzip's. Responses
include `Vary: Accept-Encoding`, a representation-specific content ETag and
`Cache-Control: public, no-cache`. Browser HTTP caches revalidate across reloads and deployments; a matching ETag returns 304.
The browser also shares the in-flight request and retains the decoded snapshot
for the page lifetime. Failed requests are not cached and the Retry button
requests the data again. Response decoding is bounded to 32 MiB with a 30-second
inactivity timeout, reset whenever another chunk arrives, and a five-minute total
transfer limit. Active downloads can therefore finish on slower connections without
allowing stalled or endless responses. Missing or malformed delivery files return a generic 503 and are logged on the
server; raw file paths and validation details are not sent to the browser.

Both build-time validation and browser reconstruction retain shared dictionary
objects instead of copying model metadata into every result. The server never
serializes the expanded snapshot into page HTML. The API serves the bundled
public summary, with no runtime Hub requests, filesystem refresh or raw-result
access. All scopes remain in the compact response, allowing language changes,
comparisons and details without additional requests. Query state is read from the
current URL when the dashboard mounts, including changes made during loading.

The loading screen respects reduced-motion settings. It cannot replace Hugging
Face's container startup screen before Next.js accepts requests.
The footer reports when results were prepared and links to their source revision;
it does not imply live synchronization.

## Implementation

| File | Responsibility |
| --- | --- |
| [publish-display.py](scripts/publish-display.py) | Latest public source snapshot, Xet download and guarded upload |
| [generate-display.ts](scripts/generate-display.ts) | Offline conversion and reduction report |
| [fetch-display.ts](scripts/fetch-display.ts) | Bounded download at an immutable Dataset SHA |
| [result-loader.ts](src/lib/result-loader.ts) | Offline result validation and source consistency |
| [display-data.ts](src/lib/display-data.ts) | Compact format, checksum, reduction checks and atomic writes |
| [prepare-delivery.ts](src/lib/prepare-delivery.ts) | Build-time validation, compression and atomic delivery manifest |
| [results.ts](src/lib/results.ts) | Cache prepared bytes and serve precomputed ETags |
| [display-client.ts](src/lib/display-client.ts) | Bounded browser fetch and dictionary reconstruction |
| [AsyncDashboard.tsx](src/components/AsyncDashboard.tsx) | Loading, retry and current-URL restoration |
| [start.ts](scripts/start.ts) | Display path and safe host selection |

The previous request-driven refresh, worker IPC, cache namespaces and two-generation
Bucket store have been removed. Legacy cache files can be discarded after stopping
older deployments. No Bucket contents are deleted automatically.
