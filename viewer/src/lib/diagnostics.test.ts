import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { prepareScoring, type ScoringCase } from './diagnostics';
import { diagnosticMean, overallIndex, METRICS, TASKS, type Snapshot, type ResultSummary } from './types';

type Fixture = {
  name: string; task: ScoringCase['questions'][number]['task']; options: ScoringCase['questions'][number]['options'];
  samples: { options?: ScoringCase['questions'][number]['options']; target: Record<string, number>; prediction: Record<string, number>; repeat: number }[];
  expected: Record<string, number | null>;
};
const fixtures: Fixture[] = JSON.parse(readFileSync('../evaluator/tests/fixtures/diagnostics.json', 'utf8'));
for (const fixture of fixtures) test(`Shared scoring: ${fixture.name}`, () => {
  const cases: ScoringCase[] = [], predictions = [];
  for (const sample of fixture.samples) for (let i = 0; i < sample.repeat; i++) {
    const case_id = String(cases.length);
    cases.push({ case_id, questions: [{ id: 'q', task: fixture.task, options: sample.options ?? fixture.options }], targets: { q: { probabilities: sample.target } } });
    predictions.push({ case_id, question_id: 'q', probabilities: sample.prediction });
  }
  const scoring = prepareScoring(cases, fixture.task);
  const metrics = scoring.calculate(predictions);
  for (const [name, expected] of Object.entries(fixture.expected)) {
    if (expected === null) assert.equal(metrics[name], null, name);
    else assert.ok(metrics[name] !== null && Math.abs(metrics[name]! - expected) < 1e-10, `${name}: ${metrics[name]} != ${expected}`);
  }
  assert.throws(() => scoring.calculate([{ case_id: 'unknown', question_id: 'q', probabilities: fixture.samples[0].prediction }]), /Unknown/);
});

test('Aggregate requires full coverage, uses dataset-wide exclusions and equal task weights', () => {
  const benchmarks = [...TASKS, 'choice' as const].map((task, i) => ({ id: String(i), task, dataset: 'x', split: 'test', case_count: 1, decision_count: 1, primary_metric: METRICS[task].name }));
  const category = { id: 'test', name: 'Test', description: '', benchmarks: benchmarks.map(b => b.id) };
  const results: ResultSummary[] = benchmarks.map((b, i) => ({
    run_id: 'run', benchmark: b, model: { id: 'm', adapter: 'm', revision: '1', settings: {} },
    created_at: '', evaluator_version: '1', provenance: 'measured', status: 'complete', elapsed_seconds: 0,
    counts: { cases: 1, expected: 1, succeeded: 1, failed: 0 },
    metrics: { [b.primary_metric]: 1, baseline_adjusted_score: [0.2, 0.4, 0.9, null][i] },
  }));
  const snapshot: Snapshot = { categories: [category], benchmarks, results, sources: [], issues: [],
    scoring: Object.fromEntries(benchmarks.map((b, i) => [b.id, { eligible: i !== 3, reason: i === 3 ? 'No headroom' : null }])) };
  assert.equal(diagnosticMean(snapshot, category, 'choice', results, 'baseline_adjusted_score'), 0.2);
  assert.equal(overallIndex(snapshot, category, results), 0.5);
  // Even a baseline-ineligible benchmark must have a complete prediction file.
  assert.equal(overallIndex(snapshot, category, results.slice(0, 3)), null);
  assert.equal(overallIndex(snapshot, category, results.map((r, i) => i === 1 ? { ...r, status: 'partial' } : r)), null);
  assert.equal(overallIndex(snapshot, category, results.map(r => ({ ...r, provenance: 'demo' }))), null);
  assert.equal(overallIndex(snapshot, category, results.map((r, i) => i === 1 ? { ...r, metrics: {} } : r)), null);
});

test('General-purpose breakdown participates in full task means without extra overall weight', async () => {
  const { generalizationCategory } = await import('./types');
  const benchmarks = TASKS.flatMap(task => ['specialized', 'diverse', 'contextual'].map(kind => ({
    id: `${kind}-${task}`, task, dataset: kind === 'specialized' ? 'datasets/other' : `datasets/s1mb-generalization-${kind}-${task}`,
    split: 'test', case_count: 100, decision_count: 100, primary_metric: METRICS[task].name,
  })));
  const category = { id: 'all', name: 'All', description: '', benchmarks: benchmarks.map(b => b.id) };
  const results: ResultSummary[] = benchmarks.map(b => ({
    run_id: 'run', benchmark: b, model: { id: 'm', adapter: 'm', revision: '1', settings: {} },
    created_at: '', evaluator_version: '1', provenance: 'measured', status: 'complete', elapsed_seconds: 0,
    counts: { cases: 100, expected: 100, succeeded: 100, failed: 0 },
    metrics: { baseline_adjusted_score: b.id.startsWith('specialized') ? 0 : 0.9 },
  }));
  const snapshot: Snapshot = { categories: [category], benchmarks, results, sources: [], issues: [],
    scoring: Object.fromEntries(benchmarks.map(b => [b.id, { eligible: true, reason: null }])) };
  const general = generalizationCategory(snapshot, category);
  assert.equal(general.benchmarks.length, 6);
  assert.ok(Math.abs(overallIndex(snapshot, category, results)! - 0.6) < 1e-10);
  for (const task of TASKS) assert.equal(diagnosticMean(snapshot, general, task, results, 'baseline_adjusted_score'), 0.9);
  const missing = results.filter(r => r.benchmark.id !== 'contextual-noul');
  assert.equal(diagnosticMean(snapshot, general, 'noul', missing, 'baseline_adjusted_score'), null);
  assert.equal(overallIndex(snapshot, category, missing), null);
  assert.equal(diagnosticMean(snapshot, general, 'choice', missing, 'baseline_adjusted_score'), 0.9);
  assert.equal(generalizationCategory(snapshot, { ...category, benchmarks: ['specialized-noul'] }).benchmarks.length, 0);
});
