import assert from 'node:assert/strict';
import { test } from 'node:test';
import { generalizationComparison } from './generalization';
import { DISPLAY_TASKS, METRICS, type Snapshot } from './types';

function fixture(): Snapshot {
  const benchmarks = ['diverse', 'contextual'].flatMap(family => DISPLAY_TASKS.map(task => ({
    id: `${family}-${task}`, task, dataset: `datasets/s1mb-generalization-${family}-${task}`,
    split: 'test', case_count: 1, decision_count: 1, primary_metric: METRICS[task].name,
  })));
  return { benchmarks, categories: [{ id: 'test', name: 'Synthetic', description: '', benchmarks: benchmarks.map(b => b.id) }], issues: [], sources: [],
    scoring: Object.fromEntries(benchmarks.map(b => [b.id, { eligible: true, reason: null }])),
    results: benchmarks.map((benchmark, i) => ({ run_id: 'synthetic', benchmark, model: { id: 'Synthetic', adapter: 'test', settings: {} }, provenance: 'measured', status: 'complete', counts: { cases: 1, expected: 1, succeeded: 1, failed: 0 },
      metrics: { baseline_adjusted_score: i / 5 },
    })),
  };
}
test('six generalization cells have fixed family/task order and an equal-weight mean including zero', () => {
  const data = fixture(), result = generalizationComparison(data, data.categories[0]);
  assert.deepEqual(result.columns.map(c => `${c.family}-${c.task}`), ['diverse-noul', 'diverse-choice', 'diverse-score', 'contextual-noul', 'contextual-choice', 'contextual-score']);
  assert.equal(result.rows[0].average, 0.5);
  assert.equal(result.rows[0].values[0], 0);
  data.results.forEach(r => { r.metrics.baseline_adjusted_score = 0; });
  assert.equal(generalizationComparison(data, data.categories[0]).rows[0].average, 0);
});
test('missing, partial, demo or undefined results keep the six-score average unavailable', () => {
  for (const state of ['missing', 'partial', 'demo', 'undefined', 'ineligible']) {
    const data = fixture();
    if (state === 'missing') data.results.shift();
    if (state === 'partial') data.results[0].status = 'partial';
    if (state === 'demo') data.results[0].provenance = 'demo';
    if (state === 'undefined') data.results[0].metrics.baseline_adjusted_score = null;
    if (state === 'ineligible') data.scoring![data.benchmarks[0].id].eligible = false;
    const row = generalizationComparison(data, data.categories[0]).rows[0];
    assert.equal(row.average, null, state);
    assert.equal(row.values[0], null, state);
    assert.equal(row.values[5], 1, state);
  }
});
test('a changed or incomplete set of definitions cannot silently reduce the average denominator', () => {
  const data = fixture();
  data.categories[0].benchmarks.pop();
  const result = generalizationComparison(data, data.categories[0]);
  assert.equal(result.columns.length, 5);
  assert.equal(result.rows[0].average, null);
});

test('six-benchmark coverage admits a model only in the selected generalization scope', async () => {
  const { hasCompleteCoverage } = await import('./comparison');
  const { generalizationCategory, overallIndex } = await import('./types');
  const data = fixture();
  const extra = { ...data.benchmarks[0], id: 'specialized', dataset: 'datasets/specialized' };
  data.benchmarks.push(extra);
  data.categories[0].benchmarks.push(extra.id);
  const full = data.categories[0];
  const general = generalizationCategory(data, full);
  assert.equal(general.benchmarks.length, 6);
  assert.equal(hasCompleteCoverage(full, data.results), false);
  assert.equal(hasCompleteCoverage(general, data.results), true);
  assert.equal(overallIndex(data, full, data.results), null);
  assert.equal(overallIndex(data, general, data.results), 0.5);
  for (const missing of general.benchmarks) {
    assert.equal(hasCompleteCoverage(general, data.results.filter(r => r.benchmark.id !== missing)), false);
  }
  data.results[0].status = 'partial';
  assert.equal(hasCompleteCoverage(general, data.results), false);
  data.results[0].status = 'complete';
  data.results[0].provenance = 'demo';
  assert.equal(hasCompleteCoverage(general, data.results), false);
  data.results[0].provenance = 'measured';
  data.results.push({ ...data.results[0], benchmark: extra });
  assert.equal(hasCompleteCoverage(full, data.results), true);
  assert.equal(hasCompleteCoverage(general, data.results), true);
});
