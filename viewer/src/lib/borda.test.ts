import assert from 'node:assert/strict';
import test from 'node:test';
import { bordaScores } from './borda';
import { generalizationCategory, METRICS, type Snapshot, type Task } from './types';

function fixture(values: number[][], tasks: Task[] = ['choice', 'noul', 'score']): Snapshot {
  const benchmarks = tasks.map((task, index) => ({ id: `benchmark-${index}`, task, dataset: 'synthetic', split: 'test', case_count: 1, decision_count: 1, primary_metric: METRICS[task].name }));
  return {
    benchmarks, categories: [{ id: 'test', name: 'Synthetic', description: '', benchmarks: benchmarks.map(b => b.id) }],
    scoring: Object.fromEntries(benchmarks.map(b => [b.id, { eligible: true, reason: null }])), sources: [], issues: [],
    results: values.flatMap((row, model) => benchmarks.map((benchmark, index) => ({
      benchmark, run_id: `${model}`, model: { id: `Synthetic ${model}`, adapter: 'test', settings: {} },
      provenance: 'measured' as const, status: 'complete' as const,
      counts: { cases: 1, expected: 1, succeeded: 1, failed: 0 }, metrics: { baseline_adjusted_score: row[index] },
    }))),
  };
}
const calculate = (snapshot: Snapshot) => bordaScores(snapshot, snapshot.categories[0]);
const close = (actual: number | null | undefined, expected: number) => assert.ok(actual != null && Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);

test('Borda uses benchmark ranks, averages ties and weights all benchmarks equally', () => {
  const snapshot = fixture([[1, 1, 0, 0.5], [0, 0, 1, 0.5], [0.5, 0.5, 0.5, 0.5]], ['choice', 'choice', 'noul', 'score']);
  const scores = calculate(snapshot);
  // Choice: 1, 0, .5; Noul: 0, 1, .5; Score: .5 for all.
  close(scores.get('0'), 0.625); close(scores.get('1'), 0.375); close(scores.get('2'), 0.5);
  const tied = calculate(fixture([[1, 1, 1], [1, 1, 1], [0, 0, 0]]));
  close(tied.get('0'), 0.75); close(tied.get('1'), 0.75); close(tied.get('2'), 0);
});

test('Score magnitudes and input order do not change ranks; roster changes can', () => {
  const snapshot = fixture([[0.9, 0.9, 0.9], [0.8, 0.8, 0.8], [0, 0, 0]]);
  close(calculate(snapshot).get('1'), 0.5);
  const shifted = { ...snapshot, results: [...snapshot.results].reverse().map(r => ({ ...r, metrics: { baseline_adjusted_score: r.metrics.baseline_adjusted_score! ** 2 } })) };
  assert.deepEqual(calculate(shifted), calculate(snapshot));
  close(calculate({ ...snapshot, results: snapshot.results.filter(r => r.run_id !== '2') }).get('1'), 0);
});

test('All 137 benchmarks contribute once without task reweighting', () => {
  const tasks: Task[] = [...Array<Task>(135).fill('choice'), 'noul', 'score'];
  const snapshot = fixture([
    [...Array<number>(135).fill(1), 0, 0],
    [...Array<number>(135).fill(0), 1, 1],
  ], tasks);
  close(calculate(snapshot).get('0'), 135 / 137);
  close(calculate(snapshot).get('1'), 2 / 137);
});

test('Generalization scope recalculates ranks and admits models complete only in that subset', () => {
  const tasks: Task[] = ['choice', 'noul', 'score', 'choice', 'noul', 'score', 'choice', 'noul', 'score'];
  const snapshot = fixture([
    [1, 1, 1, 1, 1, 1, 0.2, 0.2, 0.2],
    [0, 0, 0, 0, 0, 0, 0.8, 0.8, 0.8],
    [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5],
  ], tasks);
  snapshot.benchmarks.slice(6).forEach(b => { b.dataset = `datasets/s1mb-generalization-diverse-${b.task}`; });
  snapshot.results = snapshot.results.filter(r => r.run_id !== '2' || snapshot.benchmarks.slice(6).some(b => b.id === r.benchmark.id));
  const full = calculate(snapshot);
  close(full.get('0'), 2 / 3); close(full.get('1'), 1 / 3);
  assert.equal(full.has('2'), false);
  const general = bordaScores(snapshot, generalizationCategory(snapshot, snapshot.categories[0]));
  close(general.get('0'), 0); close(general.get('1'), 1); close(general.get('2'), 0.5);
  assert.deepEqual(calculate(snapshot), full);
});

test('Incomplete, demo and undefined adjusted scores cannot enter the cohort', () => {
  for (const failure of ['partial', 'demo', 'missing', 'undefined']) {
    const snapshot = fixture([[1, 1, 1], [0.5, 0.5, 0.5], [0, 0, 0]]);
    const result = snapshot.results[0];
    if (failure === 'partial') result.status = 'partial';
    if (failure === 'demo') result.provenance = 'demo';
    if (failure === 'missing') snapshot.results.shift();
    if (failure === 'undefined') result.metrics.baseline_adjusted_score = null;
    const scores = calculate(snapshot);
    assert.equal(scores.has('0'), false);
    close(scores.get('1'), 1); close(scores.get('2'), 0);
  }
});

test('Require two complete models and defined adjustments for every active benchmark', () => {
  assert.equal(calculate(fixture([[1, 1, 1]])).get('0'), null);
  assert.equal(calculate(fixture([])).size, 0);
  const snapshot = fixture([[1, 0, 1, 1], [0, 1, 0, 0]], ['choice', 'choice', 'noul', 'score']);
  // Recorded per-result eligibility overrides current snapshot eligibility.
  snapshot.results[0].scoring = { eligible: false, reason: 'Synthetic revision' };
  assert.equal(calculate(snapshot).get('0'), null);
  assert.equal(calculate(snapshot).get('1'), null);
  snapshot.results[5].scoring = { eligible: false, reason: 'Synthetic revision' };
  assert.equal(calculate(snapshot).get('0'), null);
  assert.equal(calculate(snapshot).get('1'), null);
});
