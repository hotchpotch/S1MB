import assert from 'node:assert/strict';
import { test } from 'node:test';
import { metadataSchema } from './result-files';
import { exposureLabel, exposuresFor } from './training-exposure';
import { encodeDisplay, decodeDisplay } from './display-data';
import { summarize } from './result-loader';

const exposure = { benchmark_id: 'choice', relationship: 'same_source_family' as const,
  dataset: 'example/train', revision: 'a'.repeat(40), config: 'nli', split: 'train',
  evidence_url: 'https://example.org/provenance', note: 'Train source family; evaluation examples not established.' };
const metadata = { model_id: 'example__model', display_name: 'Example', short_name: 'Example' };
const benchmark = { id: 'choice', task: 'choice' as const, dataset: 'datasets/example', split: 'test', case_count: 2, decision_count: 2, primary_metric: 'target_mass_at_prediction' };
const raw = { format_version: 1, run_id: 'example__model', benchmark,
  model: { id: 'example/model', adapter: 'test', settings: { secret: 'never-display' } }, provenance: 'measured', status: 'complete',
  counts: { cases: 2, expected: 2, succeeded: 2, failed: 0 },
  metrics: { target_mass_at_prediction: .8, fixed_answer_accuracy_baseline: .5, baseline_adjusted_score: .6 } };

test('declared exposure survives the bounded display artifact without changing measured scores', () => {
  const original = summarize(raw);
  const exposed = summarize({ ...raw, model: { ...raw.model, training_exposure: [exposure] } });
  const snapshot = { categories: [{ id: 'all', name: 'All', description: '', benchmarks: ['choice'] }], benchmarks: [benchmark], results: [exposed], sources: [], issues: [] };
  const decoded = decodeDisplay(encodeDisplay(snapshot, null)).snapshot;
  assert.deepEqual(decoded.results[0].model.training_exposure, [exposure]);
  assert.deepEqual(decoded.results[0].metrics, original.metrics);
  assert.deepEqual(decoded.results[0].counts, original.counts);
  assert.equal('secret' in decoded.results[0].model.settings, false);
});

test('missing disclosure is unknown, and badges apply only to named benchmarks', () => {
  assert.deepEqual(exposuresFor(raw.model), []);
  const model = { ...raw.model, training_exposure: [exposure] };
  assert.deepEqual(exposuresFor(model, 'score'), []);
  assert.equal(exposureLabel(exposuresFor(model, 'choice')), 'Known domain');
  assert.equal(exposureLabel([{ ...exposure, relationship: 'evaluation_examples' }]), 'Evaluation examples used');
});

test('reject unpinned, unbounded, secret-bearing and unexpected training metadata', () => {
  for (const bad of [{ revision: 'main' }, { evidence_url: 'https://user:secret@example.org' },
    { evidence_url: 'javascript:alert(1)' }, { split: 'x'.repeat(65) }, { predictions: [] }, { note: 'x'.repeat(1001) }, { relationship: 'unseen' }]) {
    assert.throws(() => metadataSchema.parse({ ...metadata, training_exposure: [{ ...exposure, ...bad }] }));
  }
  assert.throws(() => metadataSchema.parse({ ...metadata, training_exposure: Array(257).fill(exposure) }));
  assert.throws(() => metadataSchema.parse({ ...metadata, training_exposure: [] }));
});
