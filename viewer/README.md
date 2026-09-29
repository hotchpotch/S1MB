# S1MB viewer

Next.js / TypeScript viewer for local and synchronized Hugging Face S1MB results, with shared shadcn/ui and
Storybook components. Requires Node.js 22.22.2 and npm.

```sh
npm ci
npm run build
npm start
```

Acquire the evaluation dataset by running the evaluator first. `data` is a relative
symlink to `../evaluator/data`; do not copy or commit a second dataset. The viewer
reads Arrow directly, including Zstandard-compressed streams, without a Python
process. No data, credentials or results are required for builds or public tests.

By default, results come from synchronized `data/hub-results` when present, otherwise
`data/results`. Published model folders contain `metadata.json` and per-benchmark
`.json.xz` files. Install `xz-utils` to read compressed results. See the
[results workflow](../docs/contributing_results.md) for export, Dataset PRs and
synchronization. To compare additional local folders:

```sh
npm start -- --results-dir ./data/results --results-dir /path/to/other-results
```

`metadata.json` provides model names, links and optional parameter counts; other
JSON/XZ files must be benchmark results. Each published model folder is one
leaderboard row, including results from different runs and dataset revisions.
Published results are validated against their recorded dataset revision
(materialized by synchronization) and deduplicated by model folder and benchmark
ID. Raw local results are validated against the installed dataset and grouped by
run ID. Local sources load at startup. Restart after local results or evaluation data changes. Missing inputs,
conflicting results, and incomplete coverage are not silently assigned scores.
The browser receives summaries, never raw inputs or prediction files.

The start wrapper binds to Tailscale IPv4 when available, otherwise localhost.
Use `--host 127.0.0.1` to require localhost or `--port 3000` to select a port.
Public/all-interface binding is rejected. The startup log prints the actual URL.

Overview task scores and the overall index use a higher-is-better 0–100 adjusted
scale. Details retain primary metric directions. Only complete category coverage
qualifies for aggregate rankings. See [scoring definitions](../evaluator/SCORING.md).

## Generalization benchmarks

Under **Benchmarks → Generalization 6**, compare all six Diverse/Contextual
Noul, Choice and Score benchmarks in one table. The first numeric column is the
equal-weight mean of all six adjusted scores; it is unavailable if any required
score is missing, incomplete or undefined. Zero is a valid score and stays in
the average. The table explains why at/below-baseline scores become zero, why
zero is not zero accuracy, and how to inspect the original metrics. This section
is separate from the leaderboard, which retains its three General task columns.

## Share a comparison

The leaderboard shows only fully evaluated measured models with complete scores.
Partial results remain accessible in Compare and benchmark details.

Select models and open **Compare**. The six-axis profile shows Noul, Choice,
Score, and the same three tasks within the general-purpose subset. All axes use
the existing baseline-adjusted 0–100 scores. Missing coverage stays unavailable;
it is never plotted as zero or used to complete a polygon.

The model picker is a list of names linked to their model pages, with completed
benchmark counts. Selected models and their six scores appear to the left of a
compact landscape image: a radar on its left and model scores on its right.
Run IDs remain internal to selection URLs and are omitted from this comparison.

Use the up/down buttons beside selected models to change the chart legend and
comparison table order. Colors stay attached to models when reordered. **Copy
comparison link** preserves that order in the URL; on browsers without clipboard
access, copy the displayed URL manually. Shared links show the viewer's current
results, while downloaded images capture the scores at export time.

**Download PNG** saves the displayed profile at twice its native resolution,
including model names, numeric scores, scale notes and Hub source
provenance when available. Export runs entirely in the browser without uploading
data or requiring a screenshot service. The preview scrolls horizontally on
narrow screens; its exported image always includes the complete card.

## Development

```sh
npm run dev
npm test
npm run typecheck
npm run build
```

Dataset integration tests skip explicitly when data is absent or disabled by the
[public CI configuration](../.github/workflows/check.yml). All other tests use synthetic fixtures. Storybook is built
by `npm run build` and served at `/storybook/`; its measurements are explicitly
synthetic. `npm run build-storybook` rebuilds it separately.

## Live Hugging Face results

Set the viewer results repository option in [the environment sample](../.env.sample)
to the designated HF results Dataset ID and pass those settings as runtime
environment variables. Leave it empty for local mode. The sample documents the
revision, cache directory, minimum one-hour check interval and evaluation data
mount options. The wrapper does not automatically load the repository-root `.env`;
use your process manager or Docker's `--env-file`. Explicit local result directories
and Hub mode are mutually exclusive.

At startup, the server reuses a fresh disk cache or resolves the configured branch
(default `main`) to an exact commit SHA-1. It downloads only model `metadata.json`
and `.json.xz` files into an OS temporary directory. After the interval expires,
the first page request checks the SHA again; idle servers do not poll. An unchanged
SHA requires one small API request and no listing, download or rescoring. A changed
SHA triggers a paginated tree listing; unchanged file objects are copied from the
previous snapshot and only changed/new files are downloaded, with four transfers
at most. A page reload/navigation sees the update; an already-open browser is not
pushed new data.

Every download is pinned to that SHA and checked against Git blob SHA-1 or LFS
SHA-256 plus its declared size. JSON/XZ decoding and normal scoring validation
finish before the cache pointer and in-memory summaries switch. Removed results
are removed, while incomplete coverage remains visibly incomplete. Concurrent
requests coalesce; processes sharing the cache use a filesystem lock and persisted
check time. Cache storage keeps the current and previous successful generations;
failed stages are removed. Do not edit the managed cache manually.

The UI links the results Dataset, full results commit SHA-1, and each result file
at that exact commit. These are separate from each measurement's evaluation dataset
revision. Network or validation failures retain the last validated snapshot and
show a refresh warning. Retries wait at least the configured interval (or a longer
server retry delay). With no valid cached snapshot, startup fails. Authentication
is server-side only, and error response bodies/signed download URLs are not shown.

The viewer still needs prepared evaluation Arrow datasets to validate predictions
and compute baselines. Mount the current data plus any recorded revisions under
`result-datasets/`, prepared using `sync-results` or `validate-results
--download-datasets` as described in the [submission guide](../docs/contributing_results.md).
It does not download evaluation inputs or launch Python. A newly published result
requiring an uninstalled evaluation revision fails validation until that revision
is installed. Restart after changing evaluation data or definitions.

### Transport choice

- [HF CLI / Python SDK](https://huggingface.co/docs/huggingface_hub/guides/download)
  provide snapshot caching and Xet transfers. The CLI calls the SDK download
  helpers; adding Python/Xet is unnecessary for this viewer's compressed results.
- [JavaScript Hub SDK](https://huggingface.co/docs/huggingface.js/hub/README)
  provides Hub operations, but the application still needs its transactional
  validation and refresh policy.
- Direct HTTPS uses the Hub API with `expand=sha`, a revision-pinned tree listing
  only on changes, and pinned `resolve` downloads. This implementation minimizes
  requests in the common unchanged case and avoids per-file HEAD probes. HF
  [optimizes resolver requests](https://huggingface.co/docs/hub/rate-limits)
  separately from API requests. It does not assume undocumented conditional-GET
  support or claim a measured performance advantage over SDKs.
- [S3 compatibility](https://huggingface.co/docs/hub/storage-buckets-s3) targets
  Storage Buckets, a separate service from versioned Dataset repositories; it does
  not address this repository's commit-pinned results.

## Docker

From the repository root:

```sh
docker build -f viewer/Dockerfile -t s1mb-viewer .
docker volume create s1mb-viewer-cache
docker run --rm --init --network host \
  --env-file /absolute/path/to/viewer-runtime.env \
  --mount type=bind,src=/absolute/path/to/prepared-data,dst=/app/evaluator/data,readonly \
  --mount type=volume,src=s1mb-viewer-cache,dst=/tmp/s1mb-viewer-results \
  s1mb-viewer
```

Create the runtime environment file using the viewer options in
[the sample](../.env.sample); provide the read token only at runtime when required.
The data directory includes `benchmarks/`, `categories/`, `datasets/` and any
`result-datasets/`. The image preserves `viewer/data` as the relative symlink to
`../evaluator/data`. It includes Node.js 22.22.2 and `xz-utils`, runs as the `node`
user, and builds Next.js/Storybook without credentials or evaluation data. Ensure
the mounted data is readable by that user and a custom cache mount is writable.
The Docker build context excludes local results, private data and credentials.

The example uses Linux host networking so the existing wrapper can bind directly
to the host's Tailscale IPv4 address, or localhost when unavailable. It never binds
to all interfaces. The startup log reports the selected URL. In a bridge network,
localhost is container-local and ordinary port publication will not expose it;
use a deployment with an explicit safe network binding. The cache volume preserves
downloads and successful check times across container replacements.
