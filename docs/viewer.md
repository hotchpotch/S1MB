# Viewing the leaderboard locally

The viewer needs Node.js 22.22.2, npm and the `xz` executable (`xz-utils` on
Debian/Ubuntu). It reads saved metrics, not evaluation inputs, and does not run
models. See the [results dataset](https://huggingface.co/datasets/hotchpotch/s1mb-result)
for published measurements.

## Download and display published results

From the repository root, install the evaluator and synchronize a verified snapshot:

```sh
cd evaluator
uv sync --locked
uv run s1mb sync-results --repo-id hotchpotch/s1mb-result
```

Synchronization validates predictions against their recorded dataset revisions;
it may download evaluation data and require access to that dataset. A failed
synchronization leaves the installed snapshot unchanged. The viewer itself can
read a previously validated, fully downloaded results folder without dataset access.
Git/Xet pointer files are not usable measurements.

In another terminal, from the repository root:

```sh
cd viewer
npm ci
npm run build
npm start
```

Open the URL printed by the server (port 3000 by default). The wrapper binds to
the machine's Tailscale IPv4 address when available, otherwise localhost. To
explicitly use localhost, run `npm start -- --host 127.0.0.1` and open
<http://127.0.0.1:3000>. Do not bind local services to all interfaces.

`viewer/data` is a relative symlink to `../evaluator/data`. By default, the viewer
selects `data/hub-results` if present, otherwise `data/results`. An empty checkout
contains definitions but no measured leaderboard rows.

## Inspect your own evaluation or a Dataset PR

After building, run from `viewer/` to select a raw run explicitly:

```sh
npm start -- --results-dir ../evaluator/data/results/MY_RUN
```

Repeat `--results-dir` to compare published results and a local run:

```sh
npm start -- --results-dir ../evaluator/data/hub-results \
  --results-dir ../evaluator/data/results/MY_RUN
```

Each directory must exist. Local runs keep their run IDs; published model folders
use their metadata identity. Conflicting results reject a refresh instead of
silently overwriting another source. To preview a Dataset PR, follow the separate
snapshot instructions in [the submission guide](contributing_results.md#synchronize-the-leaderboard)
and pass that snapshot path with `--results-dir`.

## Read scores and refresh data

Overview scores are baseline-adjusted scores multiplied by 100; higher is better.
Complete coverage is required for each displayed aggregate. Details retain raw
metric directions: Choice target mass is higher-better, while Noul Brier and
Score normalized expected-value MAE are lower-better. See [scoring](../evaluator/SCORING.md).
Missing aggregates are not zero scores. The Generalization subset name does not
establish unseen-task generalization or training-data non-overlap.

To update Hub results, rerun `sync-results`, then restart the local viewer after
data/results changes. The viewer does not fetch Hub updates itself. Requests
trigger filesystem checks in a separate process; existing requests receive cached
summaries while changed files load. Failed refreshes preserve the previous cache.
Two verified display-only disk snapshots support recovery after restart.
See [cache details](../viewer/DISPLAY_DATA.md) for implementation and recovery.

If rows are missing, check the selected results directory, coverage, server logs,
and whether compressed files are real data and `xz` is installed. Validate raw
runs with `s1mb validate` or published folders with `s1mb validate-results` as
explained in the submission guide. Browser refresh alone cannot download Hub data.

## Develop the viewer

From `viewer/`, run `npm run dev -- --results-dir /absolute/path/to/results`.
Run `npm test`, `npm run typecheck`, and `npm run build` for viewer changes.
Build includes synthetic Storybook examples, available at `/storybook/` after
building. Public checks do not require datasets, tokens, models or GPUs.

See [the component reference](../viewer/README.md) for Docker and runtime options,
[developer workflow](developer_workflow.md) for source changes, and
[HF Space deployment](huggingface_space_deploy.md) for managed hosting.
