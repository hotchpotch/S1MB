# S1MB viewer

Next.js / TypeScript viewer for local S1MB results, with shared shadcn/ui and
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

By default, results come from `data/results`. To compare additional local folders:

```sh
npm start -- --results-dir ./data/results --results-dir /path/to/other-results
```

Every JSON file in a result directory must be a benchmark result. Results are
validated against the installed dataset, deduplicated by run and benchmark ID,
and loaded at startup. Restart after new results or data changes. Missing inputs,
conflicting results, and incomplete coverage are not silently assigned scores.
The browser receives summaries, never raw inputs or prediction files.

The start wrapper binds to Tailscale IPv4 when available, otherwise localhost.
Use `--host 127.0.0.1` to require localhost or `--port 3000` to select a port.
Public/all-interface binding is rejected. The startup log prints the actual URL.

Overview task scores and the overall index use a higher-is-better 0–100 adjusted
scale. Details retain primary metric directions. Only complete category coverage
qualifies for aggregate rankings. See [scoring definitions](../evaluator/SCORING.md).

## Development

```sh
npm run dev
npm test
npm run typecheck
npm run build
```

Dataset integration tests skip explicitly when data is absent, or when
`S1MB_TEST_NO_DATASET=1`. All other tests use synthetic fixtures. Storybook is built
by `npm run build` and served at `/storybook/`; its measurements are explicitly
synthetic. `npm run build-storybook` rebuilds it separately.

The project does not ship measured benchmark results or a deployment image.
When packaging a local runtime, copy the contents of the `data` symlink and keep
private datasets and results outside publicly served asset directories.
