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
Repeat `--results-dir` to combine folders. `S1MB_RESULTS_DIR` configures one folder.
`S1MB_DATA_DIR` points to tracked benchmark/category definitions and defaults to
`viewer/data` (a symlink to `../evaluator/data`). No evaluation dataset is needed.
The wrapper binds to Tailscale IPv4 when available, otherwise localhost.

## Filesystem cache

The first request waits for initial loading. Later requests return the current
in-memory snapshot immediately and can trigger a background Node child process.
It scans file size, modification/change time and inode; unchanged JSON/XZ files
are not reread. Changed files are decoded with bounded concurrency, output,
memory and time, and only compact summaries are retained. A second metadata scan
rejects changes occurring during loading. The complete candidate is published
atomically. Failed refreshes leave the last good snapshot available with a status
message. This is process-local caching; a restart performs a fresh initial load.

Local startup checks after every request by default. Set `--check-seconds 3600`
or `S1MB_RESULTS_CHECK_SECONDS=3600` to check at most hourly. Concurrent requests
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
  s1mb-viewer npm start -- --results-dir /mnt/results --port 3000
```

Both Docker targets contain Node.js and `xz`, with no Python or Arrow dependency.
Only source, UI assets and benchmark/category definitions are in the image.

## Hugging Face Spaces

See [deployment](../docs/huggingface_space_deploy.md). Attach the public results
Dataset as a read-only volume at `/mnt/results`. The managed mount handles Hub/Xet
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
