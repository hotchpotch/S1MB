import type { Task } from './types';
import type { ParameterRange } from './parameter-filters';

export type LeaderboardSort = 'borda' | 'overall' | Task | `general-${Task}`;
export type ModelOrderSort = Exclude<LeaderboardSort, 'borda'>;
export const MODEL_ORDER_SORTS: ModelOrderSort[] = ['overall', 'noul', 'choice', 'score', 'general-noul', 'general-choice', 'general-score'];
export type View = 'leaderboard' | 'benchmarks' | 'compare';
export const BENCHMARK_METRICS = {
  choice: [{ name: 'target_mass_at_prediction', label: 'Target mass' }],
  noul: [
    { name: 'f1', label: 'F1' }, { name: 'precision', label: 'Precision' },
    { name: 'positive_recall', label: 'Recall' }, { name: 'balanced_accuracy', label: 'Balanced acc.' },
    { name: 'binary_brier', label: 'Brier' },
  ],
  score: [{ name: 'normalized_expected_score_mae', label: 'MAE' }, { name: 'normalized_score_rmse', label: 'RMSE' }],
};
export type DashboardState = {
  categoryId: string; task: Task; generalizationView: boolean; view: View;
  runId: string; selectedIds: string[]; benchmarkId: string;
  generalizationOnly: boolean; checkedOnly: boolean; modelSearch: string;
  filtersOpen: boolean; totalRange: ParameterRange; activeRange: ParameterRange;
  includeIncomplete: boolean; modelPickerOpen: boolean; query: string;
  leaderboardSort: LeaderboardSort; leaderboardAscending: boolean; benchmarkMetric: string;
  comparisonQuery: string; comparisonSort: 'gap' | 'name'; comparisonDelta: boolean;
  comparisonReference: string; comparisonAdjusted: boolean;
  detailQuery: string; detailTask: 'all' | Task; trainingFocus: string;
  modelOrderSort: ModelOrderSort | null; modelOrderAscending: boolean;
};
export function defaultDashboardState(): DashboardState {
  return {
    categoryId: 'english-v1', task: 'choice', generalizationView: false, view: 'leaderboard',
    runId: '', selectedIds: [], benchmarkId: '', generalizationOnly: false, checkedOnly: false,
    modelSearch: '', filtersOpen: false, totalRange: [0, 100], activeRange: [0, 100],
    includeIncomplete: false, modelPickerOpen: true, query: '', leaderboardSort: 'borda',
    leaderboardAscending: false, benchmarkMetric: 'baseline_adjusted_score',
    comparisonQuery: '', comparisonSort: 'gap', comparisonDelta: false, comparisonReference: '',
    comparisonAdjusted: true, detailQuery: '', detailTask: 'all', trainingFocus: '', modelOrderSort: null, modelOrderAscending: false,
  };
}
const tasks = ['choice', 'noul', 'score'] as const;
function enumValue<T extends string>(value: string | null, values: readonly T[], fallback: T): T {
  return values.includes(value as T) ? value as T : fallback;
}
function parseRange(value: string | null): ParameterRange {
  const parts = value?.split(',');
  if (!parts || parts.length !== 2 || parts.some(part => !part.trim())) return [0, 100];
  const [low, high] = parts.map(Number);
  return Number.isFinite(low) && Number.isFinite(high) && low >= 0 && high <= 100 && low <= high
    ? [low, high] : [0, 100];
}
const stringFields = [
  ['modelSearch', 'modelSearch'], ['query', 'query'], ['comparisonQuery', 'comparisonQuery'],
  ['comparisonReference', 'reference'], ['detailQuery', 'detailQuery'],
] as const;
const booleanFields = [
  ['generalizationOnly', 'generalOnly'], ['checkedOnly', 'checkedOnly'], ['filtersOpen', 'filtersOpen'],
  ['modelOrderAscending', 'modelAscending'], ['includeIncomplete', 'incomplete'], ['leaderboardAscending', 'ascending'], ['comparisonDelta', 'delta'],
] as const;
export function parseDashboardState(params: URLSearchParams): DashboardState {
  const state = defaultDashboardState();
  state.categoryId = enumValue(params.get('category'), ['english-v1', 'japanese-v1'], 'english-v1');
  state.generalizationView = params.get('task') === 'generalization';
  state.task = enumValue(params.get(state.generalizationView ? 'benchmarkTask' : 'task'), tasks, 'choice');
  state.view = enumValue(params.get('view'), ['leaderboard', 'benchmarks', 'compare'], 'leaderboard');
  state.runId = params.get('run') ?? '';
  state.selectedIds = [...new Set(params.getAll('compare').filter(Boolean))];
  state.benchmarkId = params.get('benchmark') ?? '';
  state.leaderboardSort = enumValue<LeaderboardSort>(params.get('sort'), ['borda', 'overall', ...tasks, 'general-choice', 'general-noul', 'general-score'], 'borda');
  state.benchmarkMetric = enumValue(params.get('metric'), ['baseline_adjusted_score', ...BENCHMARK_METRICS[state.task].map(metric => metric.name)], 'baseline_adjusted_score');
  state.comparisonSort = enumValue(params.get('comparisonSort'), ['gap', 'name'], 'gap');
  const modelSort = params.get('modelSort');
  state.modelOrderSort = MODEL_ORDER_SORTS.includes(modelSort as ModelOrderSort) ? modelSort as ModelOrderSort : null;
  state.detailTask = enumValue(params.get('detailTask'), ['all', ...tasks], 'all');
  const training = params.get('training') ?? '';
  state.trainingFocus = state.runId && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(training) ? training : '';
  for (const [field, key] of stringFields) state[field] = params.get(key) ?? '';
  for (const [field, key] of booleanFields) state[field] = params.get(key) === '1';
  state.modelPickerOpen = params.get('pickerClosed') !== '1';
  state.comparisonAdjusted = params.get('raw') !== '1';
  state.totalRange = parseRange(params.get('totalRange'));
  state.activeRange = parseRange(params.get('activeRange'));
  return state;
}
const keys = ['category', 'task', 'benchmarkTask', 'view', 'run', 'benchmark', 'compare', 'sort', 'metric',
  'comparisonSort', 'modelSort', 'detailTask', 'training', 'totalRange', 'activeRange', 'pickerClosed', 'raw',
  ...stringFields.map(([, key]) => key), ...booleanFields.map(([, key]) => key)];
export function serializeDashboardState(state: DashboardState, base = new URLSearchParams()): URLSearchParams {
  const params = new URLSearchParams(base);
  for (const key of keys) params.delete(key);
  params.set('category', state.categoryId);
  if (state.generalizationView) {
    params.set('task', 'generalization');
    if (state.task !== 'choice') params.set('benchmarkTask', state.task);
  }
  else if (state.task !== 'choice') params.set('task', state.task);
  if (state.view !== 'leaderboard') params.set('view', state.view);
  if (state.runId) params.set('run', state.runId);
  if (state.runId && state.trainingFocus) params.set('training', state.trainingFocus);
  if (state.benchmarkId) params.set('benchmark', state.benchmarkId);
  for (const id of new Set(state.selectedIds)) if (id) params.append('compare', id);
  for (const [field, key] of stringFields) if (state[field]) params.set(key, state[field]);
  for (const [field, key] of booleanFields) if (state[field]) params.set(key, '1');
  if (state.leaderboardSort !== 'borda') params.set('sort', state.leaderboardSort);
  if (state.benchmarkMetric !== 'baseline_adjusted_score') params.set('metric', state.benchmarkMetric);
  if (state.modelOrderSort) params.set('modelSort', state.modelOrderSort);
  if (state.comparisonSort !== 'gap') params.set('comparisonSort', state.comparisonSort);
  if (state.detailTask !== 'all') params.set('detailTask', state.detailTask);
  if (!state.modelPickerOpen) params.set('pickerClosed', '1');
  if (!state.comparisonAdjusted) params.set('raw', '1');
  for (const field of ['totalRange', 'activeRange'] as const) {
    if (state[field][0] !== 0 || state[field][1] !== 100) params.set(field, state[field].join(','));
  }
  return params;
}
export function resetDashboardState(state: DashboardState): DashboardState {
  return { ...defaultDashboardState(), categoryId: state.categoryId };
}
export function selectRunDetails(state: DashboardState, runId: string, trainingFocus = ''): DashboardState {
  return { ...state, runId, trainingFocus: runId ? trainingFocus : '' };
}
type UrlBrowser = {
  parent: { postMessage(message: unknown, targetOrigin: string): void };
  location: Pick<Location, 'hostname' | 'search' | 'hash' | 'pathname'>;
  history: Pick<History, 'replaceState'>;
};
export function syncDashboardUrl(browser: UrlBrowser, params: URLSearchParams): void {
  const queryString = params.size ? `?${params}` : '';
  const { pathname, hash, hostname } = browser.location;
  browser.history.replaceState(null, '', `${pathname}${queryString}${hash}`);
  if (browser.parent !== (browser as unknown) && hostname.endsWith('.hf.space')) {
    browser.parent.postMessage({ queryString, hash }, 'https://huggingface.co');
  }
}
