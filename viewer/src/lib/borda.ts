import { categoryRuns, hasCompleteCoverage } from './comparison';
import { overallIndex, type Category, type Snapshot } from './types';

export const BORDA_DESCRIPTION = `How a model ranks against other models (0–100; higher is better).

1. Rank models by their adjusted score on each benchmark.
2. Award 100 points to first place and 0 to last, evenly spaced in between. Ties share the average points.
3. Average the points across all benchmarks, with equal weight per benchmark.

Tasks with more benchmarks have more weight. General benchmarks count once.

Only rank matters, not the size of score gaps. Adding or removing models can change this score.

Requires at least two measured models, each with complete results and valid adjusted scores for every benchmark.`;
export const TASK_AVG_DESCRIPTION = `Average performance across the three tasks (0–100; higher is better).

1. Adjust each benchmark score against its baseline and clip to 0–100.
2. Average benchmarks within each task.
3. Average Noul, Choice and Score, with one third of the weight each.

0 = at or below the baseline; 100 = the reference ceiling. This is not accuracy.

General benchmarks count once. Requires complete measured results. Other models do not affect this score.`;

/** Relative display score in [0, 1], computed against the full category cohort. */
export function bordaScores(snapshot: Snapshot, category: Category): Map<string, number | null> {
  const benchmarks = snapshot.benchmarks.filter(benchmark => category.benchmarks.includes(benchmark.id));
  const cohort = categoryRuns(snapshot, category).filter(run =>
    !run.demo && hasCompleteCoverage(category, run.results) && overallIndex(snapshot, category, run.results) !== null);
  const scores = new Map<string, number | null>(cohort.map(run => [run.id, null]));
  if (cohort.length < 2 || benchmarks.length !== category.benchmarks.length) return scores;
  const indexed = cohort.map(run => new Map(run.results.map(result => [result.benchmark.id, result])));
  // All active benchmarks must participate. Never silently drop a benchmark
  // because a recorded dataset revision makes its adjustment unavailable.
  if (benchmarks.some(benchmark => indexed.some(results => {
    const result = results.get(benchmark.id)!;
    return !(result.scoring ?? snapshot.scoring?.[benchmark.id])?.eligible ||
      !Number.isFinite(result.metrics.baseline_adjusted_score);
  }))) return scores;
  const totals = cohort.map(() => 0);
  for (const benchmark of benchmarks) {
    const ranked = indexed.map((results, index) => ({ index, value: results.get(benchmark.id)!.metrics.baseline_adjusted_score! }))
      .sort((a, b) => b.value - a.value);
    for (let start = 0; start < ranked.length;) {
      let end = start + 1;
      while (end < ranked.length && ranked[end].value === ranked[start].value) end++;
      const points = (ranked.length - 1 - (start + end - 1) / 2) / (ranked.length - 1);
      for (let position = start; position < end; position++) totals[ranked[position].index] += points;
      start = end;
    }
  }
  cohort.forEach((run, index) => { scores.set(run.id, totals[index] / benchmarks.length); });
  return scores;
}
