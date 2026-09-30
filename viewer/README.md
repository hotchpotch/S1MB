# S1MB viewer

Next.js/TypeScript leaderboard for saved S1MB metrics. Use Node.js 22.22.2, npm,
and run npm commands from this directory. Offline result conversion also needs `xz`.

```sh
npm ci
npm run fetch-display -- --revision DATASET_COMMIT_SHA
npm run dev
```

The wrapper binds to Tailscale IPv4 when available, otherwise localhost.
`data` remains a relative symlink to `../evaluator/data`. Evaluation inputs and
Python are not required to display prepared JSON.

## Runtime and deployment

- [Viewer guide](../docs/viewer.md): installation, source selection, comparing
  local and published runs, score interpretation, Docker, and troubleshooting.
- [Prepared display data](DISPLAY_DATA.md): offline conversion, reduction review,
  publication, immutable images and implementation map.
- [Space deployment](../docs/huggingface_space_deploy.md): managed hosting.
- [Scoring specification](../evaluator/SCORING.md): Task Avg and Borda formulas.
- [Configuration](../.env.sample): environment alternatives to wrapper options.

## Checks

```sh
npm test
npm run typecheck
npm run build
```

Public checks use synthetic fixtures without private data, credentials, or GPUs.
The build includes Storybook at `/storybook/`; its measurements are synthetic.
Follow [viewer guidance](AGENTS.md) for implementation and visual changes.
