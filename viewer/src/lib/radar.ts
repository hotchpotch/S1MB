import { categoryRuns, modelName } from './comparison';
import { diagnosticMean, DISPLAY_TASKS, generalizationCategory, type Category, type Snapshot } from './types';

export const RADAR_AXES = ['Noul', 'Choice', 'Score', 'General Noul', 'General Choice', 'General Score'];
const palette = ['#2563eb', '#db2777', '#059669', '#d97706', '#7c3aed', '#0891b2', '#be123c', '#4d7c0f'];

/** Keep model colors independent of selection order; never substitute zero for missing coverage. */
export function radarProfiles(snapshot: Snapshot, category: Category, selectedIds: string[]) {
  const runs = categoryRuns(snapshot, category);
  const colors = new Map(runs.map((run, index) => [run.id, palette[index] ?? `hsl(${(index * 137.508) % 360} 65% 40%)`]));
  const general = generalizationCategory(snapshot, category);
  return [...new Set(selectedIds)].flatMap(id => {
    const run = runs.find(r => r.id === id);
    if (!run) return [];
    return [{ id, name: modelName(run.model), model: run.model, color: colors.get(id)!, demo: run.demo,
      values: [category, general].flatMap(group => DISPLAY_TASKS.map(task => {
        const value = diagnosticMean(snapshot, group, task, run.results, 'baseline_adjusted_score');
        return value == null ? null : value * 100;
      })),
    }];
  });
}

export function moveComparison(ids: string[], id: string, direction: -1 | 1): string[] {
  const index = ids.indexOf(id), target = index + direction;
  if (index < 0 || target < 0 || target >= ids.length) return ids;
  const next = [...ids];
  [next[index], next[target]] = [next[target], next[index]];
  return next;
}

export function radarPoint(axis: number, value: number, radius = 140): [number, number] {
  const angle = axis * Math.PI / 3 - Math.PI / 2;
  return [285 + Math.cos(angle) * radius * value / 100, 285 + Math.sin(angle) * radius * value / 100];
}
