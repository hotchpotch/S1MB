import { categoryRuns } from './comparison';
import { diagnosticMean, DISPLAY_TASKS, type Category, type Snapshot } from './types';

/** Six individual adjusted scores; missing or undefined cells never become zero. */
export function generalizationComparison(snapshot: Snapshot, category: Category) {
  const columns = ['diverse', 'contextual'].flatMap(family => DISPLAY_TASKS.flatMap(task => {
    const benchmark = snapshot.benchmarks.find(b => category.benchmarks.includes(b.id) && b.dataset === `datasets/s1mb-generalization-${family}-${task}`);
    return benchmark ? [{ family, task, benchmark }] : [];
  }));
  const rows = categoryRuns(snapshot, category).filter(run => run.results.some(r => columns.some(c => c.benchmark.id === r.benchmark.id))).map(run => {
    const values = columns.map(({ task, benchmark }) => diagnosticMean(snapshot, { ...category, benchmarks: [benchmark.id] }, task, run.results, 'baseline_adjusted_score'));
    const average = values.length === 6 && values.every(value => value !== null)
      ? values.reduce<number>((sum, value) => sum + value!, 0) / 6 : null;
    return { ...run, values, average };
  }).sort((a, b) => Number(a.average === null) - Number(b.average === null) || (b.average ?? 0) - (a.average ?? 0) || a.id.localeCompare(b.id));
  return { columns, rows };
}
