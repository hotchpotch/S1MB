# Viewing the leaderboard locally

Use Node.js 22.22.2 and npm. The viewer reads a prepared display JSON; it requires
no Python, XZ, evaluation inputs, Hub token or network access at runtime. Offline
conversion of original results additionally requires `xz` (`xz-utils` on Debian/Ubuntu).

## Display published results

Obtain the exact public results Dataset commit SHA containing `viewer-summary.json`
from the Hub history or the maintainer's publication output. From `viewer/`:

```sh
npm ci
npm run fetch-display -- --revision DATASET_COMMIT_SHA
npm run build
npm start
```

The default artifact path is `viewer/display/viewer-summary.json`, ignored by Git.
Use `npm run dev` for development. The wrapper prints its Tailscale IPv4 or
localhost URL; local services must not listen on all interfaces.

## Convert local results

Generate the same JSON format before starting the viewer:

```sh
# From viewer/; install xz before conversion.
npm run prepare-display -- --results-dir ../evaluator/data/results/MY_RUN
npm run dev
```

Repeat `--results-dir` on the conversion command to combine existing sources:

```sh
npm run prepare-display -- --results-dir ../evaluator/data/hub-results \
  --results-dir ../evaluator/data/results/MY_RUN \
  --output ../tmp/local-viewer.json
npm start -- --display-file ../tmp/local-viewer.json
```

Local runs retain their run IDs; published model folders use their metadata IDs.
Conflicting measurements fail conversion. Fully downloaded JSON/XZ files are
required; Git/Xet pointers are not data. Definitions come from `viewer/data`,
the tracked symlink to `../evaluator/data`. `--data-dir` overrides that path on
the converter. To compare with an existing artifact, pass `--previous FILE`;
reductions block output unless its reviewed digest is explicitly approved.

For original public results, the evaluator's `sync-results` command installs a
verified snapshot; see [submission instructions](contributing_results.md).
Its validation may require access to recorded evaluation dataset revisions.
The viewer itself needs only the generated JSON. Generation from latest public
measurements and upload are described in [Space deployment](huggingface_space_deploy.md).

## Updating the display

Regenerate or fetch a new JSON, then restart the local viewer. Browsing the page
does not check the filesystem or Hub for updates. There is no background process,
Bucket cache or polling interval. The Docker image contains a fixed artifact;
rebuild and redeploy it to update the hosted leaderboard.

Runtime options for `npm start --` and `npm run dev --` are `--display-file PATH`,
`--host 127.0.0.1`, and `--port N`. Old `--results-dir`, `--cache-dir` and
`--check-seconds` startup options have been removed. Environment configuration is
listed only in [`.env.sample`](../.env.sample).

## Scores

Task scores and **Task Avg** are baseline-adjusted scores multiplied by 100;
higher is better. Task Avg weights the three tasks equally. **Borda Score** uses
relative benchmark ranks and depends on the displayed model roster. Complete
coverage in the selected scope is required. Models with all six generalization benchmarks complete appear when
**Generalization tasks only** is checked, even without other benchmark results.
The default leaderboard lists only models with complete full-category coverage.
Generalization scope does not establish unseen-task generalization.
Details retain primary metric directions. See [scoring](../evaluator/SCORING.md)
and [benchmark scope](benchmark_scope.md).

## Local Docker

Build from the repository root with the published Dataset SHA:

```sh
docker build -f viewer/Dockerfile --build-arg RESULTS_REVISION=DATASET_COMMIT_SHA \
  -t s1mb-viewer .
docker run --rm --network host s1mb-viewer
```

On Linux, host networking lets the wrapper select the host's Tailscale IPv4 or
localhost. No result or cache mounts are needed. To override with your own prepared
JSON, mount just that file read-only and select it with `--display-file`.
The image contains public display data; never bake private raw measurements or
credentials into it. Runtime is Node.js only, with no Python or XZ dependency.

## Checks and troubleshooting

From `viewer/`, run `npm test`, `npm run typecheck`, and `npm run build`.
The build includes synthetic Storybook examples at `/storybook/`. Ordinary tests
and Next.js builds are offline; the Docker bundling step deliberately fetches a
public artifact. Missing or corrupt display JSON is an error, not a silent empty
leaderboard. Check the selected file, converter report and server logs.

See [display data architecture](../viewer/DISPLAY_DATA.md) for integrity checks,
publication safeguards and recovery, and [developer workflow](developer_workflow.md)
for source contribution guidance.
