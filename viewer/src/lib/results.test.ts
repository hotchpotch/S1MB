import assert from 'node:assert/strict';
import { test } from 'node:test';
import { existsSync } from 'node:fs';
const datasetTest = !process.env.S1MB_TEST_NO_DATASET && existsSync('data/datasets/hub-source.json') ? test : test.skip;
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { loadSnapshot } from './results';
import { loadHfCases } from './hf-data';
import { prepareScoring } from './diagnostics';
import { instructionLabel, leaderboard } from './types';

const data = path.resolve('data');
async function sampleResult() {
  const benchmark = JSON.parse(await readFile(path.join(data, 'benchmarks/arc-choice-test-v1.json'), 'utf8'));
  const cases = await loadHfCases(data, benchmark.dataset, benchmark.split);
  const predictions = cases.slice(0, 2).flatMap(c => c.questions.map(q => ({ case_id: c.case_id, question_id: q.id,
    probabilities: Object.fromEntries(q.options.map(o => [o.id, 1 / q.options.length])), error: null })));
  return { format_version: 1, run_id: 'fixture', benchmark,
    model: { id: 'test', adapter: 'fixture', revision: '1', settings: { renderer: 'reviewed-system-and-instruction-v1' } },
    created_at: '2026-09-26T00:00:00Z', evaluator_version: '0.1.0', provenance: 'measured', status: 'partial',
    counts: { cases: 2, expected: benchmark.decision_count, succeeded: predictions.length, failed: 0 },
    metrics: prepareScoring(cases, 'choice').calculate(predictions), elapsed_seconds: 0,
    environment: { input_hashes: Object.fromEntries(cases.map(c => [c.case_id, c.input_hash])) }, predictions };
}

datasetTest('Changed inputs are rejected even with identical case IDs and targets', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-stale-'));
  try {
    const sample = await sampleResult();
    const id = sample.predictions[0].case_id;
    sample.environment.input_hashes[id] = 'different-input-same-case-id';
    await writeFile(path.join(root, 'stale.json'), JSON.stringify(sample));
    const snapshot = await loadSnapshot(data, [root]);
    assert.equal(snapshot.results.length, 0);
    assert.match(snapshot.issues.join(), /input hashes differ/);
  } finally { await rm(root, { recursive: true }); }
});

datasetTest('Multiple directories deduplicate results; conflicts are excluded', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-viewer-'));
  try {
    const a = path.join(root, 'a'), b = path.join(root, 'b');
    await mkdir(a); await mkdir(b);
    const sample = await sampleResult();
    await writeFile(path.join(a, 'one.json'), JSON.stringify(sample));
    await writeFile(path.join(b, 'same.json'), JSON.stringify(sample));
    let snapshot = await loadSnapshot(data, [a, b]);
    assert.equal(snapshot.results.length, 1);
    assert.equal('predictions' in snapshot.results[0], false);
    assert.equal(snapshot.issues.length, 0);
    const category = snapshot.categories.find(c => c.id === 'english-v1')!;
    assert.equal(leaderboard(snapshot, category, 'choice')[0].score, null);
    sample.created_at = '2026-09-25T10:00:00Z';
    await writeFile(path.join(b, 'same.json'), JSON.stringify(sample));
    snapshot = await loadSnapshot(data, [a, b]);
    assert.equal(snapshot.results.length, 0);
    assert.match(snapshot.issues.join(), /Conflicting/);
  } finally { await rm(root, { recursive: true }); }
});

datasetTest('Empty directory is valid; missing directories are reported', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-viewer-'));
  try {
    const snapshot = await loadSnapshot(data, [root, path.join(root, 'missing')]);
    assert.equal(snapshot.results.length, 0);
    assert.equal(snapshot.issues.length, 1);
    await assert.rejects(loadSnapshot(data, [path.join(root, 'missing')]), /No readable/);
  } finally { await rm(root, { recursive: true }); }
});

datasetTest('A partial result cannot claim a complete score', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-viewer-'));
  try {
    const sample = await sampleResult();
    sample.status = 'complete';
    await writeFile(path.join(root, 'invalid.json'), JSON.stringify(sample));
    const snapshot = await loadSnapshot(data, [root]);
    assert.equal(snapshot.results.length, 0);
    assert.match(snapshot.issues.join(), /completion/);
  } finally { await rm(root, { recursive: true }); }
});

datasetTest('Category definitions and scoring load without measured results', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-category-'));
  try {
    const snapshot = await loadSnapshot(data, [root]);
    assert.deepEqual(snapshot.issues, []);
    assert.equal(snapshot.results.length, 0);
    assert.equal(snapshot.benchmarks.length, 137);
    assert.equal(Object.keys(snapshot.scoring!).length, 137);
    for (const category of snapshot.categories) for (const task of ['choice', 'noul', 'score'] as const) {
      assert.deepEqual(leaderboard(snapshot, category, task), []);
    }
  } finally { await rm(root, { recursive: true }); }
});

datasetTest('Batch configurations of one checkpoint remain separate runs', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-instructions-'));
  try {
    const baseline = await sampleResult();
    const variant = structuredClone(baseline);
    variant.run_id = 'same-model-question-batch';
    (variant.model.settings as Record<string, unknown>).questions_per_call = 1;
    await writeFile(path.join(root, 'default.json'), JSON.stringify(baseline));
    await writeFile(path.join(root, 'variant.json'), JSON.stringify(variant));
    const snapshot = await loadSnapshot(data, [root]);
    assert.deepEqual(snapshot.issues, []);
    const rows = leaderboard(snapshot, snapshot.categories.find(c => c.id === 'english-v1')!, 'choice');
    assert.equal(rows.length, 2);
    assert.deepEqual(new Set(rows.map(r => instructionLabel(r.model))), new Set(['Dataset instructions', 'Dataset instructions · 1 question(s)/call']));
  } finally { await rm(root, { recursive: true }); }
});
