/** Version 1 descriptive scoring. Keep formulas in sync with s1mb.diagnostics. */
import type { Task } from './types';

export type ScoringCase = {
  case_id: string;
  input_hash?: string;
  questions: { id: string; task: Task; options: { id: string; value?: number | null }[] }[];
  targets: Record<string, { probabilities: Record<string, number> }>;
};
type Prediction = { case_id: string; question_id: string; probabilities: Record<string, number> | null };
type Entry = { options: ScoringCase['questions'][number]['options']; target: Record<string, number> };
export type ScoringContext = { eligible: boolean; reason: string | null };
const EPSILON = 1e-12;
const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
const key = (caseId: string, questionId: string) => JSON.stringify([caseId, questionId]);
const normalized = (entry: Entry, p: Record<string, number>) => {
  const values = entry.options.map(o => o.value!);
  const low = Math.min(...values), width = Math.max(...values) - low;
  return entry.options.reduce((sum, o) => sum + (o.value! - low) / width * p[o.id], 0);
};

export function prepareScoring(cases: ScoringCase[], task: Task) {
  const index = new Map<string, Entry>();
  for (const c of cases) for (const q of c.questions.filter(q => q.task === task)) {
    const id = key(c.case_id, q.id);
    if (index.has(id)) throw new Error('Duplicate dataset decision');
    const target = c.targets[q.id].probabilities;
    if (q.options.length < 2 || new Set(q.options.map(o => o.id)).size !== q.options.length ||
      Object.keys(target).length !== q.options.length ||
      q.options.some(o => !Number.isFinite(target[o.id]) || target[o.id] < 0 || target[o.id] > 1) ||
      Math.abs(Object.values(target).reduce((a, b) => a + b, 0) - 1) > 1e-5) throw new Error('Invalid dataset target');
    if (task === 'noul' && (q.options.length !== 2 || !('true' in target) || !('false' in target))) throw new Error('Invalid Noul options');
    if (task === 'score' && (q.options.some(o => !Number.isFinite(o.value)) || new Set(q.options.map(o => o.value)).size < 2)) throw new Error('Invalid Score scale');
    index.set(id, { options: q.options, target });
  }
  const entries = [...index.values()];
  if (!entries.length) throw new Error('Empty dataset');
  const baseline: Record<string, number> = {};
  let eligible: boolean;
  let reason: string;
  if (task === 'noul') {
    const prevalence = mean(entries.map(e => e.target.true));
    baseline.positive_prevalence = prevalence;
    baseline.majority_accuracy_baseline = Math.max(prevalence, 1 - prevalence);
    baseline.constant_brier_baseline = mean(entries.map(e => (e.target.true - prevalence) ** 2));
    eligible = prevalence > EPSILON && prevalence < 1 - EPSILON;
    reason = 'Only one target class: balanced accuracy is undefined.';
  } else if (task === 'choice') {
    const uniform = mean(entries.map(e => 1 / e.options.length));
    const ids = new Set(entries.flatMap(e => e.options.map(o => o.id)));
    const fixedId = Math.max(...[...ids].map(id => mean(entries.map(e => e.target[id] ?? 1 / e.options.length))));
    const fixedPosition = Math.max(...Array.from({ length: Math.max(...entries.map(e => e.options.length)) }, (_, i) =>
      mean(entries.map(e => i < e.options.length ? e.target[e.options[i].id] : 1 / e.options.length))));
    baseline.uniform_accuracy_baseline = uniform;
    baseline.fixed_answer_accuracy_baseline = Math.max(uniform, fixedId, fixedPosition);
    eligible = baseline.fixed_answer_accuracy_baseline < 1 - EPSILON;
    reason = 'A fixed answer already achieves 100%: no headroom for adjustment.';
  } else {
    const values = entries.map(e => normalized(e, e.target)).sort((a, b) => a - b);
    const middle = Math.floor(values.length / 2);
    const constant = values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
    baseline.constant_score_baseline = constant;
    baseline.constant_mae_baseline = mean(values.map(v => Math.abs(v - constant)));
    eligible = baseline.constant_mae_baseline > EPSILON;
    reason = 'Constant targets: baseline MAE is zero.';
  }
  const context: ScoringContext = { eligible, reason: eligible ? null : reason };
  function calculate(predictions: Prediction[]): Record<string, number | null> {
    const seen = new Set<string>();
    const rows: { entry: Entry; p: Record<string, number> }[] = [];
    for (const pred of predictions) {
      const id = key(pred.case_id, pred.question_id), entry = index.get(id);
      if (!entry || seen.has(id)) throw new Error('Unknown or duplicate prediction ID');
      seen.add(id);
      if (!pred.probabilities) continue;
      const p = pred.probabilities;
      if (Object.keys(p).length !== entry.options.length || entry.options.some(o => !(o.id in p))) throw new Error('Prediction options differ from dataset');
      rows.push({ entry, p });
    }
    const out: Record<string, number | null> = { ...baseline };
    const n = rows.length;
    let skill: number | null = null;
    if (task === 'noul') {
      let tp = 0, fp = 0, fn = 0, tn = 0;
      for (const { entry: { target: t }, p } of rows) {
        if (p.true >= 0.5) { tp += t.true; fp += t.false; }
        else { fn += t.true; tn += t.false; }
      }
      const recall = tp + fn > EPSILON ? tp / (tp + fn) : null;
      const specificity = tn + fp > EPSILON ? tn / (tn + fp) : null;
      const balanced = recall !== null && specificity !== null ? (recall + specificity) / 2 : null;
      const brier = n ? mean(rows.map(({ entry, p }) => (p.true - entry.target.true) ** 2)) : null;
      Object.assign(out, { binary_brier: brier, accuracy: n ? (tp + tn) / n : null,
        positive_recall: recall, specificity, precision: tp + fp > EPSILON ? tp / (tp + fp) : null,
        balanced_accuracy: balanced, f1: 2 * tp + fp + fn > EPSILON ? 2 * tp / (2 * tp + fp + fn) : null, true_positive: tp, false_positive: fp, false_negative: fn, true_negative: tn,
        brier_skill: baseline.constant_brier_baseline > EPSILON && brier !== null ? 1 - brier / baseline.constant_brier_baseline : null });
      skill = balanced === null ? null : 2 * balanced - 1;
    } else if (task === 'choice') {
      const accuracy = n ? mean(rows.map(({ entry, p }) => {
        const chosen = entry.options.reduce((best, o) => p[o.id] > p[best.id] ? o : best);
        return entry.target[chosen.id];
      })) : null;
      out.target_mass_at_prediction = accuracy;
      skill = eligible && accuracy !== null ? (accuracy - baseline.fixed_answer_accuracy_baseline) / (1 - baseline.fixed_answer_accuracy_baseline) : null;
    } else {
      const errors = rows.map(({ entry, p }) => {
        const values = entry.options.map(o => o.value!);
        return entry.options.reduce((sum, o) => sum + o.value! * (p[o.id] - entry.target[o.id]), 0)
          / (Math.max(...values) - Math.min(...values));
      });
      const mae = n ? mean(errors.map(Math.abs)) : null;
      out.normalized_expected_score_mae = mae;
      out.normalized_score_rmse = n ? Math.sqrt(mean(errors.map(e => e * e))) : null;
      skill = eligible && mae !== null ? 1 - mae / baseline.constant_mae_baseline : null;
    }
    out.baseline_adjusted_skill = skill;
    out.baseline_adjusted_score = skill === null ? null : Math.max(0, Math.min(1, skill));
    return out;
  }
  return { context, calculate, decisions: index.size, cases: new Set([...index.keys()].map(k => JSON.parse(k)[0])).size };
}
