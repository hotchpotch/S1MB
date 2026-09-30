export type Task = 'choice' | 'noul' | 'score';
export type Benchmark = { id: string; task: Task; dataset: string; split: string; case_count: number; decision_count: number; primary_metric: string };
export type Category = { id: string; name: string; description: string; benchmarks: string[] };
export type ModelInfo = { display_name?: string; short_name?: string; url?: string | null; hf_url?: string | null; id: string; adapter: string; settings: Record<string, unknown>; total_params?: number | null; active_params?: number | null; parameter_count_method?: "non_lookup_parameters_v1" | null };
export function instructionLabel(model: ModelInfo): string {
  const label = 'Dataset instructions';
  return typeof model.settings.questions_per_call === 'number' ? `${label} · ${model.settings.questions_per_call} question(s)/call` : label;
}
export type ResultSummary = {
  resultUrl?: string; evaluator_original_run_id?: string; dataset_source?: { repo_id: string; revision: string }; scoring?: { eligible: boolean; reason: string | null };
  run_id: string; benchmark: Benchmark; model: ModelInfo;
  provenance: 'measured' | 'demo'; status: 'complete' | 'partial';
  counts: { cases: number; expected: number; succeeded: number; failed: number };
  metrics: Record<string, number | null>;
};
export type Snapshot = { display?: { generatedAt: string; source: { repo: string; revision: string } | null; digest: string }; scoring?: Record<string, { eligible: boolean; reason: string | null }>; categories: Category[]; benchmarks: Benchmark[]; results: ResultSummary[]; issues: string[]; sources: { name: string; files: number }[] };
export const TASKS: Task[] = ['choice', 'noul', 'score'];
export const METRICS: Record<Task, { name: string; label: string; direction: 'up' | 'down'; description: string }> = {
  choice: { name: 'target_mass_at_prediction', label: 'Target mass at prediction', direction: 'up', description: 'Target probability assigned to the selected option. Equal to accuracy for hard labels.' },
  noul: { name: 'binary_brier', label: 'Binary Brier', direction: 'down', description: 'Mean squared error between predicted and target probability of true.' },
  score: { name: 'normalized_expected_score_mae', label: 'Normalized score MAE', direction: 'down', description: 'Expected-score absolute error, divided by the width of the original scale.' },
};

export type LeaderboardRow = { runId: string; model: ModelInfo; demo: boolean; score: number | null; complete: number; total: number; results: ResultSummary[] };
export function leaderboard(snapshot: Snapshot, category: Category, task: Task): LeaderboardRow[] {
  const required = snapshot.benchmarks.filter(b => b.task === task && category.benchmarks.includes(b.id));
  const ids = new Set(required.map(b => b.id));
  const runs = new Map<string, ResultSummary[]>();
  for (const result of snapshot.results) {
    if (!category.benchmarks.includes(result.benchmark.id)) continue;
    const group = runs.get(result.run_id) ?? [];
    group.push(result);
    runs.set(result.run_id, group);
  }
  return [...runs].map(([runId, results]) => {
    const relevant = results.filter(r => ids.has(r.benchmark.id));
    const complete = relevant.filter(r => r.status === 'complete' && r.metrics[METRICS[task].name] !== null);
    const score = required.length > 0 && complete.length === required.length
      ? complete.reduce((sum, r) => sum + r.metrics[METRICS[task].name]!, 0) / required.length : null;
    return { runId, model: results[0].model, demo: results.some(r => r.provenance === 'demo'), score, complete: complete.length, total: required.length, results };
  }).sort((a, b) => Number(a.demo) - Number(b.demo) ||
    Number(a.score === null) - Number(b.score === null) ||
    (a.score !== null && b.score !== null ? (a.score - b.score) * (task === 'choice' ? -1 : 1) : 0) ||
    a.runId.localeCompare(b.runId));
}

export const DIAGNOSTIC_COLUMNS: Record<Task, { name: string; label: string }[]> = {
  noul: [
    { name: 'accuracy', label: 'Accuracy' },
    { name: 'balanced_accuracy', label: 'Balanced acc.' },
    { name: 'positive_recall', label: 'True recall' },
    { name: 'specificity', label: 'False recall' },
  ],
  choice: [{ name: 'fixed_answer_accuracy_baseline', label: 'Fixed baseline' }],
  score: [
    { name: 'normalized_score_rmse', label: 'RMSE' },
    { name: 'constant_mae_baseline', label: 'Baseline MAE' },
  ],
};

/** Equal benchmark weights; never average over model-specific missing results. */
export function diagnosticMean(snapshot: Snapshot, category: Category, task: Task, results: ResultSummary[], metric: string): number | null {
  const required = snapshot.benchmarks.filter(b => category.benchmarks.includes(b.id) && b.task === task);
  if (!required.length || required.some(b => !results.some(r => r.benchmark.id === b.id && r.status === 'complete' && r.provenance === 'measured'))) return null;
  const eligible = metric === 'baseline_adjusted_score'
    ? required.filter(b => (results.find(r => r.benchmark.id === b.id)?.scoring ?? snapshot.scoring?.[b.id])?.eligible) : required;
  if (!eligible.length) return null;
  const values = eligible.map(b => results.find(r => r.benchmark.id === b.id)!.metrics[metric]);
  if (values.some(v => v == null)) return null;
  return values.reduce<number>((sum, v) => sum + v!, 0) / values.length;
}

/** Experimental index: equal task weights after equal benchmark weights. */
export function overallIndex(snapshot: Snapshot, category: Category, results: ResultSummary[]): number | null {
  const values = TASKS.map(task => diagnosticMean(snapshot, category, task, results, 'baseline_adjusted_score'));
  return values.every(v => v !== null) ? values.reduce<number>((sum, v) => sum + v!, 0) / values.length : null;
}

/** General-purpose tasks are a subset of the full category, not extra ranking weight. */
export function generalizationCategory(snapshot: Snapshot, category: Category): Category {
  const ids = new Set(snapshot.benchmarks.filter(b => b.dataset.startsWith('datasets/s1mb-generalization-')).map(b => b.id));
  return { ...category, benchmarks: category.benchmarks.filter(id => ids.has(id)) };
}

export const DISPLAY_TASKS: Task[] = ['noul', 'choice', 'score'];
