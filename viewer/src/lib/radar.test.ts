import assert from 'node:assert/strict';
import { test } from 'node:test';
import { moveComparison, radarProfiles } from './radar';
import { compareRuns } from './comparison';
import { DISPLAY_TASKS, METRICS, type Snapshot } from './types';

function fixture(): Snapshot {
  const benchmarks = ['specialized', 'generalization'].flatMap(group => DISPLAY_TASKS.map(task => ({
    id: `${group}-${task}`, task, dataset: `datasets/s1mb-${group}-${task}`, split: 'test', case_count: 1, decision_count: 1, primary_metric: METRICS[task].name,
  })));
  return {
    categories: [{ id: 'test', name: 'Synthetic', description: '', benchmarks: benchmarks.map(b => b.id) }],
    benchmarks, issues: [], sources: [],
    scoring: Object.fromEntries(benchmarks.map(b => [b.id, { eligible: true, reason: null }])),
    results: ['A', 'B', 'C'].flatMap(run_id => benchmarks.map((benchmark, index) => ({
      run_id, benchmark, model: { id: run_id, adapter: 'test', settings: {} }, provenance: 'measured', status: 'complete', counts: { cases: 1, expected: 1, succeeded: 1, failed: 0 },
      metrics: { baseline_adjusted_score: [0, 0.2, 0.4, 0.6, 0.8, 1][index] },
    }))),
  };
}
test('radar axes use full-category then general-subset scores in Noul, Choice, Score order', () => {
  const data = fixture();
  const profiles = radarProfiles(data, data.categories[0], ['A']);
  assert.deepEqual(profiles[0].values.map(v => Math.round(v!)), [30, 50, 70, 60, 80, 100]);
  data.results.find(r => r.run_id === 'A' && r.benchmark.id === 'generalization-noul')!.metrics.baseline_adjusted_score = 0;
  assert.equal(radarProfiles(data, data.categories[0], ['A'])[0].values[3], 0);
});
test('missing coverage remains unavailable, including its full task, without hiding other axes', () => {
  const data = fixture();
  data.results = data.results.filter(r => !(r.run_id === 'A' && r.benchmark.id === 'generalization-noul'));
  assert.deepEqual(radarProfiles(data, data.categories[0], ['A'])[0].values.map(v => v == null ? null : Math.round(v)), [null, 50, 70, null, 80, 100]);
  data.results.forEach(r => { if (r.run_id === 'A') r.provenance = 'demo'; });
  assert.deepEqual(radarProfiles(data, data.categories[0], ['A'])[0].values, Array(6).fill(null));
});
test('model reordering matches tables and URL order while retaining distinct model colors', () => {
  const data = fixture(), category = data.categories[0];
  const first = radarProfiles(data, category, ['A', 'B', 'C']);
  const order = moveComparison(moveComparison(['A', 'B', 'C'], 'A', 1), 'A', 1);
  assert.deepEqual(order, ['B', 'C', 'A']);
  const profiles = radarProfiles(data, category, order);
  assert.deepEqual(profiles.map(p => p.id), compareRuns(data, category, order).runs.map(r => r.id));
  assert.equal(new Set(first.map(p => p.color)).size, 3);
  for (const profile of profiles) assert.equal(profile.color, first.find(p => p.id === profile.id)!.color);
  assert.deepEqual(moveComparison(order, 'B', -1), order);
  assert.deepEqual(moveComparison(order, 'A', 1), order);
  assert.deepEqual(moveComparison(order, 'unknown', 1), order);
  assert.deepEqual(radarProfiles(data, category, ['unknown', 'A', 'A']).map(p => p.id), ['A']);
});
