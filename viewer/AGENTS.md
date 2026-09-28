# Viewer guidance

Use Node.js 22.22.2 and run npm commands from this directory. Run `npm test`,
`npm run typecheck` and `npm run build` after changes. Builds include Storybook
and must not require private data or credentials. Dataset integration tests skip
when data is absent or `S1MB_TEST_NO_DATASET=1`; other tests use synthetic fixtures.

Use shared shadcn/ui components. Clearly label all Storybook measurements as
synthetic. Read current compact Arrow datasets with Apache Arrow JS and native
Zstandard decoding; do not introduce historical-schema adapters or Python services.
`data` links to `../evaluator/data`; never duplicate or commit downloaded data.

Results load at startup. Restart after data or result updates. Reject conflicting
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
