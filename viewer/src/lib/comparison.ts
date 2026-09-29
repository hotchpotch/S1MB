import {
  generalizationCategory,
  METRICS,
  TASKS,
  type Benchmark,
  type Category,
  type ModelInfo,
  type ResultSummary,
  type Snapshot,
  type Task,
} from "./types";

export type Run = {
  id: string;
  model: ModelInfo;
  demo: boolean;
  results: ResultSummary[];
};
export function categoryRuns(snapshot: Snapshot, category: Category): Run[] {
  const grouped = new Map<string, ResultSummary[]>();
  const ids = new Set(category.benchmarks);
  for (const r of snapshot.results) {
    if (!ids.has(r.benchmark.id)) continue;
    grouped.set(r.run_id, [...(grouped.get(r.run_id) ?? []), r]);
  }
  return [...grouped]
    .map(([id, results]) => ({
      id,
      model: results[0].model,
      demo: results.some((r) => r.provenance === "demo"),
      results,
    }))
    .sort(
      (a, b) =>
        Number(a.demo) - Number(b.demo) ||
        a.model.id.localeCompare(b.model.id) ||
        a.id.localeCompare(b.id),
    );
}
export type ComparisonRow = {
  benchmark: Benchmark;
  cells: {
    runId: string;
    result?: ResultSummary;
    value: number | null;
    best: boolean;
  }[];
};
export function compareRuns(
  snapshot: Snapshot,
  category: Category,
  selected: string[],
  adjusted = false,
): { runs: Run[]; groups: { task: Task; rows: ComparisonRow[] }[] } {
  const available = categoryRuns(snapshot, category);
  const runs = [...new Set(selected)].flatMap((id) => {
    const run = available.find((r) => r.id === id);
    return run ? [run] : [];
  });
  const required = new Set(category.benchmarks);
  const groups = TASKS.map((task) => ({
    task,
    rows: snapshot.benchmarks
      .filter((b) => required.has(b.id) && b.task === task)
      .sort((a, b) => a.id.localeCompare(b.id))
      .map((benchmark) => {
        const cells = runs.map((run) => {
          const result = run.results.find(
            (r) => r.benchmark.id === benchmark.id,
          );
          return {
            runId: run.id,
            result,
            value: result?.metrics[adjusted ? 'baseline_adjusted_score' : benchmark.primary_metric] ?? null,
            best: false,
          };
        });
        const eligible = cells.filter(
          (c) =>
            c.value !== null &&
            c.result?.status === "complete" &&
            c.result.provenance !== "demo",
        );
        if (eligible.length > 1) {
          const best = (adjusted || METRICS[task].direction === "up" ? Math.max : Math.min)(
            ...eligible.map((c) => c.value!),
          );
          for (const cell of eligible) cell.best = cell.value === best;
        }
        return { benchmark, cells };
      }),
  }));
  return { runs, groups };
}
export function modelName(model: ModelInfo): string {
  if (model.display_name) return model.display_name;
  if (model.adapter === "system-ichi") {
    const parts = model.id.split("/").filter(Boolean);
    return (
      "System Ichi · " +
      (parts.at(-1) === "checkpoint"
        ? (parts.at(-2) ?? model.id)
        : (parts.at(-1) ?? model.id))
    );
  }
  return model.id.replace("convaiinnovations/", "");
}
export function benchmarkName(benchmark: Benchmark): string {
  return benchmark.id
    .replace(new RegExp(`-${benchmark.task}-test-v\\d+$`), "")
    .replaceAll("-", " ");
}

/** Rank a benchmark by its selected metric, keeping partial and missing values last. */
export function compareResultMetric(a: ResultSummary, b: ResultSummary, metric: string): number {
  const complete = (r: ResultSummary) => r.status === 'complete' && r.provenance === 'measured';
  const av = a.metrics[metric], bv = b.metrics[metric];
  const lower = ['binary_brier', 'normalized_expected_score_mae', 'normalized_score_rmse'].includes(metric);
  return Number(complete(b)) - Number(complete(a)) || Number(av == null) - Number(bv == null)
    || (av != null && bv != null ? (av - bv) * (lower ? 1 : -1) : 0)
    || a.run_id.localeCompare(b.run_id);
}

/** Range among complete measured cells; an absent comparison is not a tie. */
export function comparisonSpread(row: ComparisonRow): number | null {
  const values = row.cells.filter(c => c.value != null && c.result?.status === 'complete' && c.result.provenance === 'measured').map(c => c.value!);
  return values.length < 2 ? null : Math.max(...values) - Math.min(...values);
}

/** Positive means better than the reference, including lower-is-better raw metrics. */
export function comparisonDelta(cell: ComparisonRow["cells"][number], reference: ComparisonRow["cells"][number] | undefined, higherIsBetter: boolean): number | null {
  const valid = (c: ComparisonRow["cells"][number] | undefined) => c?.value != null && c.result?.status === 'complete' && c.result.provenance === 'measured';
  if (!valid(cell) || !valid(reference)) return null;
  return (cell.value! - reference!.value!) * (higherIsBetter ? 1 : -1);
}

/** Leaderboards require every active benchmark to be complete and measured. */
export function hasCompleteCoverage(category: Category, results: ResultSummary[]): boolean {
  return category.benchmarks.length > 0 && category.benchmarks.every(id =>
    results.some(result => result.benchmark.id === id && result.status === 'complete' && result.provenance === 'measured'));
}

/** Competition ranks exclude unavailable values and give ties the same rank. */
export function metricRanks(entries: { id: string; value: number | null }[], lowerIsBetter = false): Map<string, number> {
  const scored = entries.filter((entry): entry is { id: string; value: number } => entry.value != null && Number.isFinite(entry.value))
    .sort((a, b) => (a.value - b.value) * (lowerIsBetter ? 1 : -1));
  let rank = 0;
  return new Map(scored.map((entry, index) => {
    if (index === 0 || entry.value !== scored[index - 1].value) rank = index + 1;
    return [entry.id, rank];
  }));
}

/** Listing eligibility does not relax coverage for any displayed aggregate. */
export function hasLeaderboardCoverage(snapshot: Snapshot, category: Category, results: ResultSummary[]): boolean {
  return hasCompleteCoverage(category, results) ||
    hasCompleteCoverage(generalizationCategory(snapshot, category), results);
}
