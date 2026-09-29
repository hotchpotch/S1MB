import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { HubClient, HubResultsCache, hubResultsOptions, type HubResultsOptions } from './hub-results';
import type { Snapshot } from './types';

const a = 'a'.repeat(40), b = 'b'.repeat(40), c = 'c'.repeat(40);
const model = 'test__model';
const metadataPath = `${model}/metadata.json`;
const resultPath = `${model}/example.json.xz`;
const metadata = JSON.stringify({ model_id: model, display_name: 'Synthetic', short_name: 'Synthetic' });
const options = (cacheDir: string): HubResultsOptions => ({ repoId: 'test/results', revision: 'main', cacheDir, ttlMs: 3600_000, token: 'synthetic-token' });
function mockHub(config: HubResultsOptions) {
  const versions = new Map([[a, new Map([[metadataPath, metadata], [resultPath, 'synthetic compressed bytes']])]]);
  let revision = a;
  const requests: string[] = [];
  let failStatus = 0;
  let corrupt = false;
  const fetcher: typeof fetch = async (input, init) => {
    const url = new URL(String(input));
    requests.push(url.pathname + url.search);
    assert.equal((init?.headers as Record<string, string>).Authorization, 'Bearer synthetic-token');
    if (failStatus) return new Response('secret error body', { status: failStatus, headers: { 'retry-after': '7200' } });
    if (url.pathname.includes('/revision/')) return Response.json({ sha: revision });
    const pinned = url.pathname.split('/')[url.pathname.includes('/tree/') ? 6 : 5];
    const files = versions.get(pinned)!;
    assert.ok(files, 'requests must be pinned to an available commit');
    if (url.pathname.includes('/tree/')) {
      const entries = [...files].map(([name, bytes]) => ({ type: 'file', path: name, size: Buffer.byteLength(bytes),
        oid: createHash('sha1').update(`blob ${Buffer.byteLength(bytes)}\0`).update(bytes).digest('hex'),
        ...(name.endsWith('.xz') ? { lfs: { oid: createHash('sha256').update(bytes).digest('hex'), size: Buffer.byteLength(bytes) } } : {}),
      }));
      // Exercise real pagination; a card is deliberately excluded from downloads.
      return url.searchParams.has('cursor') ? Response.json(entries.slice(1)) : Response.json([
        { type: 'file', path: 'README.md' }, ...entries.slice(0, 1),
      ], { headers: { Link: `<https://huggingface.co${url.pathname}?cursor=next>; rel="next"` } });
    }
    return new Response(corrupt ? 'corrupt' : files.get(url.pathname.split('/').slice(6).join('/')));
  };
  return { client: new HubClient(config, fetcher), requests, versions,
    setRevision(value: string) { revision = value; }, fail(status: number) { failStatus = status; }, corrupt(value: boolean) { corrupt = value; } };
}
async function fixtureLoad(directory: string): Promise<Snapshot> {
  const files = (await readdir(path.join(directory, model))).filter(f => f.endsWith('.xz'));
  const results = files.map(f => ({ run_id: model, benchmark: { id: f.replace('.json.xz', '') } })) as Snapshot['results'];
  return { categories: [], benchmarks: [], results, issues: [], sources: [{ name: path.basename(directory), files: files.length }] };
}

test('Hub mode is opt-in and rejects unsafe repository IDs and intervals below one hour', () => {
  assert.equal(hubResultsOptions({}), undefined);
  const config = hubResultsOptions({ S1MB_HF_RESULTS_REPO: 'test/results' })!;
  assert.equal(config.ttlMs, 3600_000);
  assert.equal(config.revision, 'main');
  for (const value of ['0', '3599', 'NaN', 'Infinity', '3600.5', '']) assert.throws(() => hubResultsOptions({ S1MB_HF_RESULTS_REPO: 'test/results', S1MB_HF_RESULTS_CACHE_SECONDS: value }));
  assert.throws(() => hubResultsOptions({ S1MB_HF_RESULTS_REPO: '../results' }));
});

test('TTL, pinned downloads, restart cache, unchanged SHA, delta downloads and deletions', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-hub-test-'));
  try {
    const config = options(root), hub = mockHub(config);
    let time = 10_000_000, loads = 0;
    const load = async (dir: string) => { loads++; return fixtureLoad(dir); };
    const cache = new HubResultsCache(config, load, hub.client, () => time);
    const [first, concurrent] = await Promise.all([cache.get(), cache.get()]);
    assert.equal(first, concurrent);
    assert.equal(first.resultsSource?.revision, a);
    assert.equal(first.results[0].resultUrl, `https://huggingface.co/datasets/test/results/blob/${a}/${resultPath}`);
    assert.equal(hub.requests.length, 5); // SHA + two tree pages + two downloads.
    time += config.ttlMs - 1;
    await cache.get();
    assert.equal(hub.requests.length, 5);
    const restarted = new HubResultsCache(config, load, hub.client, () => time);
    await restarted.get();
    assert.equal(hub.requests.length, 5);
    time++;
    await cache.get();
    assert.equal(hub.requests.length, 6);
    assert.equal(loads, 2); // No parsing/scoring again for an unchanged SHA.
    hub.versions.set(b, new Map([[metadataPath, metadata], [resultPath, 'updated bytes'], [`${model}/second.json.xz`, 'added bytes']]));
    hub.setRevision(b);
    time += config.ttlMs;
    const changed = await cache.get();
    assert.equal(changed.resultsSource?.revision, b);
    assert.equal(changed.results.length, 2);
    assert.equal(hub.requests.filter(r => r.includes(`/resolve/${b}/`)).length, 2);
    assert.ok(!hub.requests.some(r => r.includes(`/resolve/${b}/${metadataPath}`)));
    hub.versions.set(c, new Map([[metadataPath, metadata], [resultPath, 'updated bytes']]));
    hub.setRevision(c);
    time += config.ttlMs;
    assert.equal((await cache.get()).results.length, 1);
    assert.equal(hub.requests.filter(r => r.includes(`/resolve/${c}/`)).length, 0);
    const cacheRoot = path.join(root, (await readdir(root))[0]);
    assert.equal((await readdir(cacheRoot)).filter(n => n.startsWith('snapshot-')).length, 2);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('invalid updates and rate limits preserve the installed generation and bound retries', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-hub-test-'));
  try {
    const config = options(root), hub = mockHub(config);
    let time = 10_000_000, invalid = false;
    const cache = new HubResultsCache(config, async dir => ({ ...await fixtureLoad(dir), issues: invalid ? ['Invalid synthetic input hashes'] : [] }), hub.client, () => time);
    await cache.get();
    const stateFile = path.join(root, (await readdir(root))[0], 'state.json');
    const original = await readFile(stateFile, 'utf8');
    hub.versions.set(b, new Map([[metadataPath, metadata], [resultPath, 'updated bytes']]));
    hub.setRevision(b);
    invalid = true;
    time += config.ttlMs;
    const failed = await cache.get();
    assert.equal(failed.resultsSource?.revision, a);
    assert.equal(failed.resultsSource?.refreshFailed, true);
    assert.equal(await readFile(stateFile, 'utf8'), original);
    assert.equal((await readdir(path.dirname(stateFile))).filter(n => n.startsWith('snapshot-')).length, 1);
    const count = hub.requests.length;
    await cache.get();
    assert.equal(hub.requests.length, count);
    invalid = false;
    hub.fail(429);
    time += config.ttlMs;
    await cache.get();
    time += config.ttlMs;
    await cache.get();
    assert.equal(hub.requests.length, count + 1); // Retry-After is longer than the TTL.
    hub.fail(0);
    time += config.ttlMs;
    assert.equal((await cache.get()).resultsSource?.revision, b);
    assert.equal((await cache.get()).resultsSource?.refreshFailed, undefined);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('failed first downloads never install a snapshot, and can be retried', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-hub-test-'));
  try {
    const config = options(root), hub = mockHub(config);
    const cache = new HubResultsCache(config, fixtureLoad, hub.client);
    hub.corrupt(true);
    await assert.rejects(cache.get(), /size|checksum/);
    const cacheRoot = path.join(root, (await readdir(root))[0]);
    assert.deepEqual(await readdir(cacheRoot), []);
    hub.corrupt(false);
    assert.equal((await cache.get()).resultsSource?.revision, a);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('independent caches sharing a volume serialize the initial download', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-hub-test-'));
  try {
    const config = options(root), hub = mockHub(config);
    const first = new HubResultsCache(config, fixtureLoad, hub.client);
    const second = new HubResultsCache(config, fixtureLoad, hub.client);
    const snapshots = await Promise.all([first.get(), second.get()]);
    assert.equal(snapshots[0].resultsSource?.revision, snapshots[1].resultsSource?.revision);
    assert.equal(hub.requests.filter(r => r.includes('/revision/')).length, 1);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('malformed trees, missing metadata, unsafe pagination and secret error bodies are rejected', async () => {
  const config = options('/unused');
  for (const entries of [
    [{ type: 'file', path: '../escape.json' }],
    [{ type: 'file', path: resultPath, size: 1, oid: a }],
    [{ type: 'file', path: metadataPath, size: 64 * 1024 * 1024 + 1, oid: a }],
  ]) {
    const client = new HubClient(config, async () => Response.json(entries));
    await assert.rejects(client.files(a));
  }
  const client = new HubClient(config, async () => Response.json([], { headers: { link: '<https://evil.example/steal>; rel="next"' } }));
  await assert.rejects(client.files(a), /pagination/);
  const denied = new HubClient(config, async () => new Response('secret token', { status: 401 }));
  await assert.rejects(denied.revision(), error => String(error).includes('401') && !String(error).includes('secret'));
});
