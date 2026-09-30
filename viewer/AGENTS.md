# Viewer guidance

Use Node.js 22.22.2 and run npm commands from this directory. Run `npm test`,
`npm run typecheck` and `npm run build` after implementation changes. Builds
include Storybook and must not require private data or credentials. Dataset integration tests skip
when data is absent or disabled by public CI; other tests use synthetic fixtures.

For visual changes, add or update synthetic Storybook states before changing the
design. Review primary screens at desktop and mobile widths, including selected
comparison runs and the details dialog. Keep screenshots under ignored `../tmp/`.

Use shared shadcn/ui components. Clearly label all Storybook measurements as
synthetic. The offline converter reads saved metrics from result JSON/XZ files; do not download evaluation
inputs, load Arrow, or run Python in the viewer. Publication validation belongs to
the evaluator. Check metric ranges, saved baseline adjustment, counts, membership
and display metadata before exposing a candidate snapshot.
`data` links to `../evaluator/data`; never commit downloaded data.

Follow `../docs/contributing_results.md` for publication layout. Published folders
supply row identity and display metadata. Preserve raw run identity for local runs.
Keep complete coverage requirements and per-result baseline eligibility.

Use one prepared display format for local startup and Docker. Runtime loads a local
JSON once; do not add source scans, worker IPC, mounted Dataset/Bucket dependencies
or background refresh. Offline conversion validates a full snapshot. Publication
compares against the last published artifact and stops for human approval when
measurements or coverage decrease. Preserve atomic writes, bounded decoding,
immutable source SHAs and parent-commit checks. Never serve raw inputs, predictions,
full environment settings or credentials to the browser.

The start wrapper binds to Tailscale IPv4 or localhost and rejects public/all-interface
addresses. Verify the actual listening address when starting a service.

<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
