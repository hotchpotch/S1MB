import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, mkdir, writeFile, rm, unlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { FilesystemResults, summarize } from './result-loader';
import { readResultJson } from './result-files';
import { SnapshotCache, ResultsWorker } from './results';
import type { Snapshot } from './types';

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

test('filesystem cache decodes only changed files, handles deletions and preserves state on invalid updates', async () => {
  const f = await fixture(); let reads = 0;
  const loader = new FilesystemResults(f.root, [f.results], async file => { reads++; return readResultJson(file); });
  try {
    const first = (await loader.refresh())!;
    assert.equal(first.results[0].run_id, 'example__model');
    assert.equal(first.results[0].model.display_name, 'Synthetic');
    assert.ok(!JSON.stringify(first).includes('never-render'));
    assert.ok(!JSON.stringify(first).includes('dataset_source'));
    const initialReads = reads;
    assert.equal(await loader.refresh(), null); assert.equal(reads, initialReads);
    await f.write(0.9);
    assert.equal((await loader.refresh())!.results[0].metrics.target_mass_at_prediction, 0.9);
    assert.equal(reads, initialReads + 1);
    await writeFile(f.file, 'invalid xz');
    await assert.rejects(loader.refresh(), /Invalid result/);
    await f.write(0.7);
    assert.equal((await loader.refresh())!.results[0].metrics.target_mass_at_prediction, 0.7);
    await unlink(f.file);
    assert.equal((await loader.refresh())!.results.length, 0);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});

test('background cache returns stale data immediately, coalesces requests, throttles, swaps and keeps data on failure', async () => {
  let time = 0, calls = 0;
  let resolve!: (s: Snapshot | null) => void, reject!: (e: Error) => void;
  const cache = new SnapshotCache(() => { calls++; return new Promise((yes, no) => { resolve = yes; reject = no; }); }, 3600000, () => time);
  const initial = cache.get(); assert.equal(cache.get(), initial); assert.equal(calls, 1);
  const snapshot: Snapshot = { benchmarks: [], categories: [], results: [], sources: [], issues: [] };
  resolve(snapshot); const first = await initial;
  assert.equal(await cache.get(), first); assert.equal(calls, 1);
  time = 3600000;
  assert.equal(await cache.get(), first); assert.equal(await cache.get(), first); assert.equal(calls, 2);
  resolve({ ...snapshot, issues: ['new'] }); await new Promise(r => setImmediate(r));
  const updated = await cache.get(); assert.deepEqual(updated.issues, ['new']);
  time += 3600000;
  assert.equal(await cache.get(), updated); reject(new Error('synthetic failure')); await new Promise(r => setImmediate(r));
  assert.deepEqual((await cache.get()).issues, ['new']); assert.equal((await cache.get()).cache?.refreshFailed, true);
});

test('real child process reads XZ, stays cached and refreshes without Python or evaluation data', async () => {
  const f = await fixture(); const worker = new ResultsWorker(f.root, [f.results]);
  try {
    assert.equal((await worker.refresh())!.results.length, 1);
    assert.equal(await worker.refresh(), null);
    await f.write(0.6);
    assert.equal((await worker.refresh())!.results[0].metrics.target_mass_at_prediction, 0.6);
  } finally { worker.close(); await rm(f.root, { recursive: true, force: true }); }
});

test('reject invalid scores, counts and parameter metadata; preserve incomplete results', () => {
  assert.throws(() => summarize({ ...result(), metrics: { ...result().metrics, baseline_adjusted_score: 1 } }));
  assert.throws(() => summarize({ ...result(), counts: { cases: 2, expected: 3, succeeded: 2, failed: 0 } }));
  assert.throws(() => summarize({ ...result(), model: { ...result().model, active_params: 3, total_params: 2 } }));
  const partial = summarize({ ...result(), status: 'partial', counts: { cases: 1, expected: 2, succeeded: 1, failed: 0 } });
  assert.equal(partial.status, 'partial');
});

test('zero interval checks on each access but never blocks a cached response', async () => {
  let calls = 0, resolve!: (value: Snapshot | null) => void;
  const initial: Snapshot = { benchmarks: [], categories: [], results: [], sources: [], issues: [] };
  const cache = new SnapshotCache(() => { calls++; return calls === 1 ? Promise.resolve(initial) : new Promise(r => { resolve = r; }); }, 0);
  const first = await cache.get();
  assert.equal(await cache.get(), first); assert.equal(calls, 2);
  assert.equal(await cache.get(), first); assert.equal(calls, 2);
  resolve(null); await new Promise(r => setImmediate(r));
  await cache.get(); assert.equal(calls, 3); resolve(null);
});

test('a failed initial load can retry; a candidate changing during loading is not installed', async () => {
  const f = await fixture(); let mutate = false;
  const loader = new FilesystemResults(f.root, [f.results], async file => {
    const value = await readResultJson(file);
    if (mutate && file === f.file) { mutate = false; await f.write(0.9); }
    return value;
  });
  try {
    let fail = true;
    const cache = new SnapshotCache(() => fail ? Promise.reject(new Error('synthetic initial failure')) : loader.refresh(), 0);
    await assert.rejects(cache.get(), /initial failure/); fail = false;
    assert.equal((await cache.get()).results.length, 1);
    await f.write(0.7); mutate = true;
    await assert.rejects(loader.refresh(), /changed during refresh/);
    assert.equal((await loader.refresh())!.results[0].metrics.target_mass_at_prediction, 0.9);
    const metadata = path.join(path.dirname(f.file), 'metadata.json');
    await writeFile(metadata, JSON.stringify({ model_id: 'example__model', display_name: 'Renamed synthetic', short_name: 'Renamed' }));
    assert.equal((await loader.refresh())!.results[0].model.display_name, 'Renamed synthetic');
  } finally { await rm(f.root, { recursive: true, force: true }); }
});

test('restored data is returned before the mandatory full background refresh, even inside the interval', async () => {
  const snapshot: Snapshot = { benchmarks: [], categories: [], results: [], sources: [], issues: ['restored'] };
  let calls = 0, finish!: (value: Snapshot | null) => void;
  const cache = new SnapshotCache(() => { calls++; return new Promise(r => { finish = r; }); }, 3600000, Date.now, async () => snapshot);
  assert.equal(await cache.get(), snapshot);
  assert.equal(calls, 1);
  assert.equal(await cache.get(), snapshot);
  finish({ ...snapshot, issues: ['fresh'] }); await new Promise(r => setImmediate(r));
  assert.deepEqual((await cache.get()).issues, ['fresh']);
});

test('worker restart restores JSON without accessing unavailable original files and then rebuilds from source', async () => {
  const { rename, readdir } = await import('node:fs/promises');
  const f = await fixture(), cacheDir = path.join(f.root, 'cache');
  let worker = new ResultsWorker(f.root, [f.results], cacheDir);
  try {
    const original = (await worker.refresh())!;
    worker.close();
    await rename(f.results, f.results + '-offline');
    worker = new ResultsWorker(f.root, [f.results], cacheDir);
    assert.deepEqual((await worker.restore())!.results, original.results);
    await assert.rejects(worker.refresh());
    await rename(f.results + '-offline', f.results);
    await f.write(0.9);
    assert.equal((await worker.refresh())!.results[0].metrics.target_mass_at_prediction, 0.9);
    worker.close(); worker = new ResultsWorker(f.root, [f.results], cacheDir);
    assert.equal((await worker.restore())!.results[0].metrics.target_mass_at_prediction, 0.9);
    await worker.refresh();
    const [namespace] = await readdir(cacheDir);
    assert.equal((await readdir(path.join(cacheDir, namespace))).length, 2, 'Unchanged restart must not add a generation');
  } finally { worker.close(); await rm(f.root, { recursive: true, force: true }); }
});

test('disk cache keeps two valid generations, restores fallback, cleans interrupted writes, and isolates sources', async () => {
  const { SnapshotStore } = await import('./snapshot-store');
  const { readdir, readFile } = await import('node:fs/promises');
  const f = await fixture(), cacheDir = path.join(f.root, 'cache');
  const store = new SnapshotStore(cacheDir, f.root, [f.results]);
  const loader = new FilesystemResults(f.root, [f.results]);
  try {
    assert.equal(await store.restore(), null);
    let snapshot = (await loader.refresh())!;
    await store.save(snapshot);
    const names = () => readdir(store.directory);
    const first = await names();
    await store.save({ ...snapshot, cache: { checkedAt: new Date().toISOString(), refreshFailed: false } });
    assert.deepEqual(await names(), first);
    for (const score of [0.7, 0.6, 0.9]) {
      await f.write(score); snapshot = (await loader.refresh())!; await store.save(snapshot);
      assert.equal((await names()).length, 2);
    }
    const latest = (await names()).sort().at(-1)!;
    const original = await readFile(path.join(store.directory, latest), 'utf8');
    await writeFile(path.join(store.directory, latest), '{interrupted');
    assert.equal((await store.restore())!.results[0].metrics.target_mass_at_prediction, 0.6);
    await store.save(snapshot);
    assert.equal((await names()).length, 2);
    assert.equal((await store.restore())!.results[0].metrics.target_mass_at_prediction, 0.9);
    const changed = JSON.parse(original); changed.snapshot.results[0].metrics.target_mass_at_prediction = 0.1;
    const newest = (await names()).sort().at(-1)!;
    await writeFile(path.join(store.directory, newest), JSON.stringify(changed));
    assert.equal((await store.restore())!.results[0].metrics.target_mass_at_prediction, 0.6, 'Checksum mismatch falls back');
    changed.version = 999;
    await writeFile(path.join(store.directory, newest), JSON.stringify(changed));
    assert.equal((await store.restore())!.results[0].metrics.target_mass_at_prediction, 0.6, 'Unknown format falls back');
    assert.equal(await new SnapshotStore(cacheDir, f.root, [f.results + '-other']).restore(), null);
    await store.save(snapshot);
    assert.equal((await names()).length, 2);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});

test('cache write failure does not discard fresh measurements and retries an unchanged snapshot', async () => {
  const f = await fixture(), cacheDir = path.join(f.root, 'cache');
  const { readdir } = await import('node:fs/promises');
  await writeFile(cacheDir, 'not a directory');
  const worker = new ResultsWorker(f.root, [f.results], cacheDir);
  try {
    assert.equal((await worker.refresh())!.results.length, 1);
    await unlink(cacheDir);
    assert.equal(await worker.refresh(), null);
    assert.equal((await readdir(cacheDir)).length, 1);
    assert.equal((await worker.restore())!.results.length, 1);
  } finally { worker.close(); await rm(f.root, { recursive: true, force: true }); }
});
