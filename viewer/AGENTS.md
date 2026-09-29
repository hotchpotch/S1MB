# Viewer guidance

Use Node.js 22.22.2 and run npm commands from this directory. Run `npm test`,
`npm run typecheck` and `npm run build` after implementation changes. Builds
include Storybook and must not require private data or credentials. Dataset integration tests skip
when data is absent or disabled by public CI; other tests use synthetic fixtures.

For visual changes, add or update synthetic Storybook states before changing the
design. Review primary screens at desktop and mobile widths, including selected
comparison runs and the details dialog. Keep screenshots under ignored `../tmp/`.

Use shared shadcn/ui components. Clearly label all Storybook measurements as
synthetic. Read current compact Arrow datasets with Apache Arrow JS and native
Zstandard decoding; do not introduce historical-schema adapters or Python services.
`data` links to `../evaluator/data`; never duplicate or commit downloaded data.

Follow `../docs/contributing_results.md` for result layout and synchronization.
The default source is `data/hub-results` when present, otherwise `data/results`;
explicit `--results-dir` options select the sources to load. Decode `.json.xz`
with bounded output and memory; `xz` is a runtime/test prerequisite.

For published results, use the model folder as the row identity and read display
metadata from `metadata.json`. Preserve per-benchmark original run IDs and
model/dataset/evaluator revisions in details. Mixed revisions are allowed; validate
against the recorded dataset revision materialized by synchronization and use its
baseline in aggregation. Raw local runs retain their existing run-ID semantics.

Local results load at startup. Restart after local data or result updates.
Opt-in Hub results use a temporary, validated snapshot cache and request-driven
refreshes after the configured interval; preserve atomic replacement and failure
fallback. Evaluation data and definition changes still require a restart. Reject conflicting
results and invalid input hashes. Require complete coverage for ranked aggregates,
keep missing results visible, and preserve run selection in comparison URLs.
Do not serve raw inputs, predictions or credentials as public assets.

The start wrapper binds to Tailscale IPv4 or localhost and rejects public/all-interface
addresses. Verify the actual listening address when starting a service.

<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
