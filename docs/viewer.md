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

## Share and restore view state

The viewer stores its controls in query parameters. Reloading or opening a shared
URL restores the selected scope, tab, model details, rankings, filters and ordered
comparison selection. Default values are omitted to keep links compact.

| Parameters | State |
| --- | --- |
| `category` | `english-v1` or `japanese-v1` |
| `view`, `task`, `benchmark` | Main tab, task and benchmark selection |
| `run` | Open model details sidebar |
| `compare` (repeated) | Selected comparison models in display order |
| `sort`, `ascending`, `metric` | Leaderboard column/direction and benchmark ranking metric |
| `modelSearch`, `query`, `generalOnly`, `checkedOnly` | Model/benchmark searches and leaderboard filters |
| `totalRange`, `activeRange`, `filtersOpen` | Parameter slider positions (0–100) and filter panel visibility |
| `comparisonQuery`, `comparisonSort`, `delta`, `reference`, `raw` | Comparison filtering, row order, differences, reference model and original metrics |
| `modelSort`, `modelAscending` | Selected-model ordering column and direction |
| `incomplete`, `pickerClosed` | Incomplete comparison models and model picker visibility |
| `detailQuery`, `detailTask` | Search and task filter inside model details |
| `benchmarkTask` | Underlying task retained while `task=generalization` is selected |

Boolean flags use `1`; ranges use two comma-separated slider positions. Invalid
control values fall back to defaults. Model and benchmark IDs must exist in the
selected scope for their corresponding content to appear.

Click **S1MB.** to reset controls and close model details while retaining the
current language scope. Comparison section anchors are cleared on reset.

In the hosted [Hugging Face Space](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard),
the iframe also synchronizes query parameters and anchors with the outer Space
URL using [Hugging Face's documented messaging interface](https://huggingface.co/docs/hub/spaces-handle-url-parameters).
The Space forwards its initial URL parameters to the app, so links such as
[Japanese v1](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard?category=japanese-v1)
can be shared directly.
