import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, mkdir, writeFile, rm, unlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { FilesystemResults, summarize } from './result-loader';
import { readResultJson } from './result-files';

const benchmark = { id: 'choice', task: 'choice', dataset: 'datasets/test', split: 'test', case_count: 2, decision_count: 2, primary_metric: 'target_mass_at_prediction' };
const result = (score = 0.8) => ({ format_version: 1, run_id: 'run', benchmark,
  model: { id: 'example/model', adapter: 'test', revision: 'ignored', settings: { secret: 'never-render', questions_per_call: 1 } },
  provenance: 'measured', status: 'complete', counts: { cases: 2, expected: 2, succeeded: 2, failed: 0 },
  metrics: { target_mass_at_prediction: score, fixed_answer_accuracy_baseline: 0.5, baseline_adjusted_score: Math.max(0, 2 * score - 1) },
  predictions: [{ private: 'never-render' }], dataset_source: { repo_id: 'private', revision: 'ignored' },
});
async function fixture() {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-files-'));
  const results = path.join(root, 'results'), model = path.join(results, 'example__model');
  await Promise.all(['benchmarks', 'categories', 'results/example__model'].map(d => mkdir(path.join(root, d), { recursive: true })));
  await writeFile(path.join(root, 'benchmarks/choice.json'), JSON.stringify(benchmark));
  await writeFile(path.join(root, 'categories/all.json'), JSON.stringify({ id: 'all', name: 'All', description: '', benchmarks: ['choice'] }));
  await writeFile(path.join(model, 'metadata.json'), JSON.stringify({ model_id: 'example__model', display_name: 'Synthetic', short_name: 'Synthetic' }));
  const file = path.join(model, 'choice.json.xz');
  const write = async (score: number) => writeFile(file, execFileSync('xz', ['--compress', '--stdout'], { input: JSON.stringify(result(score)) }));
  await write(0.8);
  return { root, results, file, write };
}

test('reject invalid scores, counts and parameter metadata; preserve incomplete results', () => {
  assert.throws(() => summarize({ ...result(), metrics: { ...result().metrics, baseline_adjusted_score: 1 } }));
  assert.throws(() => summarize({ ...result(), counts: { cases: 2, expected: 3, succeeded: 2, failed: 0 } }));
  assert.throws(() => summarize({ ...result(), model: { ...result().model, active_params: 3, total_params: 2 } }));
  const partial = summarize({ ...result(), status: 'partial', counts: { cases: 1, expected: 2, succeeded: 1, failed: 0 } });
  assert.equal(partial.status, 'partial');
});

test('converts native joint-question results while rejecting invalid call settings', () => {
  const joint = { ...result(), model: { ...result().model, adapter: 'clef', settings: { questions_per_call: 'all-case-questions', secret: 'never-render' } } };
  const summary = summarize(joint);
  assert.deepEqual(summary.model.settings, { questions_per_call: 'all-case-questions' });
  for (const value of [0, -1, 1.5, 'unknown']) {
    assert.throws(() => summarize({ ...joint, model: { ...joint.model, settings: { questions_per_call: value } } }));
  }
});

test('combines published and local rows without merging model runs or overriding conflicts', async () => {
  const f = await fixture();
  const local = path.join(f.root, 'local');
  await mkdir(local);
  const file = path.join(local, 'choice.json');
  await writeFile(file, JSON.stringify(result(0.9)));
  const loader = new FilesystemResults(f.root, [f.results, local]);
  try {
    const snapshot = (await loader.refresh())!;
    assert.deepEqual(snapshot.results.map(r => r.run_id).sort(), ['example__model', 'run']);
    assert.deepEqual(snapshot.sources.map(s => s.files), [1, 1]);
    await writeFile(path.join(local, 'duplicate.json'), JSON.stringify(result(0.7)));
    await assert.rejects(loader.refresh(), /Conflicting result/);
    await unlink(path.join(local, 'duplicate.json'));
    assert.equal((await loader.refresh()).results.length, 2);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});

test('offline conversion ignores display JSON and rejects invalid source files', async () => {
  const f = await fixture();
  try {
    await writeFile(path.join(f.results, 'viewer-summary.json'), '{not a result}');
    const snapshot = await new FilesystemResults(f.root, [f.results]).refresh();
    assert.equal(snapshot.results.length, 1);
    assert.ok(!JSON.stringify(snapshot).includes('never-render'));
    await writeFile(f.file, 'invalid xz');
    await assert.rejects(new FilesystemResults(f.root, [f.results]).refresh(), /Invalid result/);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});

test('compact display roundtrip, integrity, reduction guard and atomic publication candidate', async () => {
  const { encodeDisplay, decodeDisplay, atomicJson, readDisplay, reductionWarnings } = await import('./display-data');
  const { readFile } = await import('node:fs/promises');
  const { spawnSync } = await import('node:child_process');
  const f = await fixture();
  try {
    const snapshot = await new FilesystemResults(f.root, [f.results]).refresh();
    const source = { repo: 'example/results', revision: 'a'.repeat(40) };
    const artifact = encodeDisplay(snapshot, source);
    assert.deepEqual(decodeDisplay(artifact).snapshot.results, snapshot.results);
    assert.equal(artifact.payload.models.length, 1);
    const invalid = structuredClone(artifact); invalid.payload.results[0].metrics.baseline_adjusted_score = 0;
    assert.throws(() => decodeDisplay(invalid), /checksum/);
    assert.throws(() => decodeDisplay({ ...artifact, version: 999 }));
    const previous = path.join(f.root, 'previous.json');
    // Simulate a previously published additional row that is missing from the new sources.
    const larger = { ...snapshot, results: [...snapshot.results, { ...snapshot.results[0], run_id: 'removed-model' }] };
    await atomicJson(previous, encodeDisplay(larger, source));
    assert.ok(reductionWarnings(larger, snapshot).some(w => w.includes('removed-model')));
    const partial = { ...snapshot, results: [{ ...snapshot.results[0], status: 'partial' as const }] };
    assert.ok(reductionWarnings(snapshot, partial).some(w => w.includes('Completion regressed')));
    const output = path.join(f.root, 'output.json'); await writeFile(output, 'keep previous output');
    const args = ['--import', 'tsx', 'scripts/generate-display.ts', '--results-dir', f.results, '--data-dir', f.root, '--output', output, '--previous', previous];
    const blocked = spawnSync(process.execPath, args, { encoding: 'utf8' });
    assert.equal(blocked.status, 2, blocked.stderr);
    assert.equal(await readFile(output, 'utf8'), 'keep previous output');
    const report = JSON.parse(await readFile(output + '.report.json', 'utf8'));
    assert.equal(report.current.results, 1);
    const approved = spawnSync(process.execPath, [...args, '--approve-reduction', report.digest], { encoding: 'utf8' });
    assert.equal(approved.status, 0, approved.stderr);
    assert.equal((await readDisplay(output)).snapshot.results.length, 1);
    await f.write(0.9);
    assert.equal(spawnSync(process.execPath, [...args, '--approve-reduction', report.digest]).status, 2, 'Approval must match the candidate');
    await assert.rejects(atomicJson(output, 'x'.repeat(33 * 1024 * 1024)), /size limit/);
    assert.equal((await readDisplay(output)).snapshot.results.length, 1);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});

test('conversion refuses files edited while reading', async () => {
  const f = await fixture(); let changed = false;
  try {
    const loader = new FilesystemResults(f.root, [f.results], async file => {
      const value = await readResultJson(file);
      if (file === f.file && !changed) { changed = true; await f.write(0.9); }
      return value;
    });
    await assert.rejects(loader.refresh(), /changed during refresh/);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});
