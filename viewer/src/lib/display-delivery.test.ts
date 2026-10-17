import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, writeFile, rm, unlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { encodeDisplay, decodeDisplay } from './display-data';
import { expandDisplay, createDisplayLoader } from './display-client';
import { createDisplayStore, displayResponse } from './results';
import { prepareDelivery } from './prepare-delivery';
import type { Snapshot } from './types';

function fixture() {
  const benchmark = { id: 'choice', task: 'choice' as const, dataset: 'datasets/test', split: 'test', case_count: 2, decision_count: 2, primary_metric: 'target_mass_at_prediction' };
  const model = { id: 'test/model', adapter: 'test', settings: { questions_per_call: 1 } };
  const result = { run_id: 'test__model', benchmark, model, provenance: 'measured' as const, status: 'complete' as const,
    counts: { cases: 2, expected: 2, succeeded: 2, failed: 0 }, metrics: { target_mass_at_prediction: 0.8, fixed_answer_accuracy_baseline: 0.5, baseline_adjusted_score: 0.6 }, scoring: { eligible: true, reason: null } };
  const second = { ...benchmark, id: 'second' };
  const snapshot: Snapshot = { benchmarks: [benchmark, second], categories: [{ id: 'english-v1', name: 'English', description: '', benchmarks: ['choice', 'second'] }],
    results: [result, { ...result, benchmark: second }], sources: [{ name: 'local', files: 2 }], issues: [] };
  return { ...encodeDisplay(snapshot, null), generatedAt: '2026-10-13T00:00:00.000Z' };
}

test('server and browser expansion preserve scores and shared dictionary objects', () => {
  const artifact = fixture();
  const server = decodeDisplay(artifact).snapshot;
  const browser = expandDisplay(artifact);
  assert.deepEqual(browser, server);
  assert.equal(server.results[0].model, server.results[1].model);
  assert.equal(browser.results[0].model, browser.results[1].model);
  assert.equal(browser.results[0].benchmark, browser.benchmarks[0]);
  assert.equal(server.results[0].benchmark, server.benchmarks[0]);
  const invalid = structuredClone(artifact); invalid.payload.results[0].model = 999;
  assert.throws(() => expandDisplay(invalid), /reference/);
});

test('compact delivery caches once and revalidates with ETag', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-delivery-'));
  const file = path.join(root, 'display.json');
  const artifact = fixture();
  await writeFile(file, JSON.stringify(artifact));
  await prepareDelivery(file);
  const store = createDisplayStore(file);
  try {
    const [a, b] = await Promise.all([store(), store()]);
    assert.equal(a, b);
    await unlink(file);
    const response = await displayResponse(new Request('http://localhost/api/display/'), store);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), artifact);
    assert.match(response.headers.get('cache-control')!, /no-cache/);
    const cached = await displayResponse(new Request('http://localhost/api/display/', { headers: { 'If-None-Match': response.headers.get('etag')! } }), store);
    assert.equal(cached.status, 304);
    assert.equal(await cached.text(), '');
    assert.equal(await store(), a);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('invalid local data returns a safe failure and a corrected file can retry', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-delivery-'));
  const file = path.join(root, 'display.json');
  await writeFile(file, '{}');
  const store = createDisplayStore(file);
  try {
    const failed = await displayResponse(new Request('http://localhost/api/display/'), store);
    assert.equal(failed.status, 503);
    assert.match(failed.headers.get('cache-control')!, /no-store/);
    assert.ok(!(await failed.text()).includes(file));
    await writeFile(file, JSON.stringify(fixture()));
    await prepareDelivery(file);
    assert.equal((await displayResponse(new Request('http://localhost/api/display/'), store)).status, 200);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('browser shares requests, retains loaded data and retries failed requests', async () => {
  let calls = 0;
  const load = createDisplayLoader(async () => { calls++; return Response.json(fixture()); });
  const [a, b] = await Promise.all([load(), load()]);
  assert.equal(a, b);
  assert.equal(await load(), a);
  assert.equal(calls, 1);
  let attempts = 0;
  const retry = createDisplayLoader(async () => ++attempts === 1 ? new Response('unavailable', { status: 503 }) : Response.json(fixture()));
  await assert.rejects(retry(), /503/);
  assert.deepEqual(await retry(), expandDisplay(fixture()));
  assert.equal(attempts, 2);
});

test('browser rejects malformed and oversized responses', async () => {
  await assert.rejects(createDisplayLoader(async () => Response.json({ version: 99 }))(), /display/i);
  await assert.rejects(createDisplayLoader(async () => new Response('large', { headers: { 'content-length': '40000000' } }))(), /size/i);
});

test('ETag covers generation and source metadata even when scores are unchanged', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-delivery-'));
  const file = path.join(root, 'display.json');
  const a = fixture();
  try {
    await writeFile(file, JSON.stringify(a));
    await prepareDelivery(file);
    const first = await createDisplayStore(file)();
    await writeFile(file, JSON.stringify({ ...a, generatedAt: '2026-10-14T00:00:00.000Z' }));
    await prepareDelivery(file);
    const second = await createDisplayStore(file)();
    assert.notEqual(first.etag, second.etag);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('browser bounds chunked responses even without a Content-Length header', async () => {
  let canceled = false;
  const load = createDisplayLoader(async () => new Response(new ReadableStream({
    pull(controller) { controller.enqueue(new Uint8Array(1024 * 1024)); },
    cancel() { canceled = true; },
  })));
  await assert.rejects(load(), /size/i);
  assert.equal(canceled, true);
});

test('compact validation still rejects inconsistent scores, counts and references', () => {
  // Recompute the digest so these exercise semantic validation rather than checksum rejection.
  const rehash = (artifact: ReturnType<typeof fixture>) => {
    artifact.digest = createHash('sha256').update(JSON.stringify(artifact.payload)).digest('hex');
    return artifact;
  };
  const score = fixture(); score.payload.results[0].metrics.baseline_adjusted_score = 0;
  assert.throws(() => decodeDisplay(rehash(score)), /adjusted score/);
  const count = fixture(); count.payload.results[0].counts.expected = 3;
  assert.throws(() => decodeDisplay(rehash(count)), /counts/);
  const definition = fixture(); definition.payload.definitions[0] = 999;
  assert.throws(() => decodeDisplay(rehash(definition)), /reference/);
});

test('initial page renders a small loading shell without a display file or result props', async () => {
  const { renderToStaticMarkup } = await import('react-dom/server');
  const { createElement } = await import('react');
  const { default: Page } = await import('../app/page');
  const html = renderToStaticMarkup(createElement(Page));
  assert.match(html, /Preparing the leaderboard/);
  assert.ok(Buffer.byteLength(html) < 16_000, 'Initial shell must remain independent of result volume');
  assert.ok(!html.includes('baseline_adjusted_score'));
});

test('compact API compresses negotiated responses and keeps representation ETags distinct', async () => {
  const { gunzipSync, brotliDecompressSync } = await import('node:zlib');
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-delivery-'));
  const file = path.join(root, 'display.json');
  const artifact = fixture();
  await writeFile(file, JSON.stringify(artifact));
  await prepareDelivery(file);
  const store = createDisplayStore(file);
  try {
    const plain = await displayResponse(new Request('http://localhost/api/display/'), store);
    const zipped = await displayResponse(new Request('http://localhost/api/display/', { headers: { 'accept-encoding': 'gzip' } }), store);
    assert.equal(zipped.headers.get('content-encoding'), 'gzip');
    assert.equal(zipped.headers.get('vary'), 'Accept-Encoding');
    assert.notEqual(zipped.headers.get('etag'), plain.headers.get('etag'));
    assert.deepEqual(JSON.parse(gunzipSync(Buffer.from(await zipped.arrayBuffer())).toString()), artifact);
    const brotli = await displayResponse(new Request('http://localhost/api/display/', { headers: { 'accept-encoding': 'gzip, deflate, br' } }), store);
    assert.equal(brotli.headers.get('content-encoding'), 'br');
    assert.notEqual(brotli.headers.get('etag'), zipped.headers.get('etag'));
    const brotliBytes = Buffer.from(await brotli.arrayBuffer());
    assert.equal(Number(brotli.headers.get('content-length')), brotliBytes.length);
    assert.deepEqual(JSON.parse(brotliDecompressSync(brotliBytes).toString()), artifact);
    const preferGzip = await displayResponse(new Request('http://localhost/api/display/', { headers: { 'accept-encoding': 'br;q=0.5, gzip' } }), store);
    assert.equal(preferGzip.headers.get('content-encoding'), 'gzip');
    const noBrotli = await displayResponse(new Request('http://localhost/api/display/', { headers: { 'accept-encoding': 'br;q=0, gzip' } }), store);
    assert.equal(noBrotli.headers.get('content-encoding'), 'gzip');
    const disabled = await displayResponse(new Request('http://localhost/api/display/', { headers: { 'accept-encoding': 'gzip;q=0' } }), store);
    assert.equal(disabled.headers.get('content-encoding'), null);
    const cached = await displayResponse(new Request('http://localhost/api/display/', { headers: { 'accept-encoding': 'gzip', 'if-none-match': zipped.headers.get('etag')! } }), store);
    assert.equal(cached.status, 304);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('build preparation rejects invalid input without replacing a working delivery', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-build-'));
  const file = path.join(root, 'display.json');
  try {
    await writeFile(file, JSON.stringify(fixture()));
    await prepareDelivery(file);
    await unlink(file);
    const before = await createDisplayStore(file)();
    await writeFile(file, '{}');
    await assert.rejects(prepareDelivery(file));
    const after = await createDisplayStore(file)();
    assert.deepEqual(after, before);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('runtime bounds manifests and rejects paths or truncated prepared bytes', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-manifest-'));
  const file = path.join(root, 'display.json');
  const manifest = path.join(`${file}.delivery`, 'manifest.json');
  try {
    await writeFile(file, JSON.stringify(fixture()));
    await prepareDelivery(file);
    await writeFile(manifest, JSON.stringify({ version: 1, digest: '../../private' }));
    await assert.rejects(createDisplayStore(file)(), /manifest/);
    await writeFile(manifest, ' '.repeat(1025));
    await assert.rejects(createDisplayStore(file)(), /size/);
    await prepareDelivery(file);
    const { readFile } = await import('node:fs/promises');
    const metadata = JSON.parse(await readFile(manifest, 'utf8'));
    await writeFile(path.join(`${file}.delivery`, `${metadata.digest}.gz`), 'truncated');
    await assert.rejects(createDisplayStore(file)(), /Incomplete/);
  } finally { await rm(root, { recursive: true, force: true }); }
});


test('existing display artifacts without abstention counts retain their checksum', () => {
  const artifact = fixture();
  for (const result of artifact.payload.results) delete result.counts.abstained;
  artifact.digest = createHash('sha256').update(JSON.stringify(artifact.payload)).digest('hex');
  const decoded = decodeDisplay(artifact);
  assert.equal(decoded.artifact.digest, artifact.digest);
  assert.equal('abstained' in decoded.artifact.payload.results[0].counts, false);
  assert.equal(decoded.snapshot.results[0].status, 'complete');
});

test('display artifacts preserve and validate explicit abstention counts', () => {
  const artifact = fixture();
  for (const result of artifact.payload.results) {
    result.counts.succeeded = 1;
    result.counts.abstained = 1;
    result.metrics.target_mass_at_prediction = 0.5;
    result.metrics.baseline_adjusted_score = 0;
  }
  artifact.digest = createHash('sha256').update(JSON.stringify(artifact.payload)).digest('hex');
  const decoded = decodeDisplay(artifact);
  assert.equal(decoded.snapshot.results[0].counts.abstained, 1);
  assert.equal(decoded.snapshot.results[0].status, 'complete');
  const invalid = structuredClone(artifact);
  invalid.payload.results[0].counts.abstained = 2;
  invalid.digest = createHash('sha256').update(JSON.stringify(invalid.payload)).digest('hex');
  assert.throws(() => decodeDisplay(invalid), /counts/);
});


test('browser allows an active transfer beyond the idle timeout but bounds stalled and endless transfers', async () => {
  const bytes = new TextEncoder().encode(JSON.stringify(fixture()));
  const request = (stall: boolean): typeof fetch => async (_url, init) => {
    const signal = init!.signal!;
    let offset = 0;
    let canceled = false;
    return new Response(new ReadableStream({
      start(controller) {
        signal.addEventListener('abort', () => controller.error(signal.reason), { once: true });
      },
      async pull(controller) {
        await new Promise(resolve => setTimeout(resolve, 20));
        if (signal.aborted || canceled) return;
        if (stall) return new Promise(() => {});
        if (offset === bytes.length) { controller.close(); return; }
        const end = Math.min(offset + Math.ceil(bytes.length / 10), bytes.length);
        controller.enqueue(bytes.slice(offset, end));
        offset = end;
      },
      cancel() { canceled = true; },
    }));
  };
  assert.deepEqual(await createDisplayLoader(request(false), 100, 2_000)(), expandDisplay(fixture()));
  await assert.rejects(createDisplayLoader(request(true), 50, 2_000)(), /stalled/);
  await assert.rejects(createDisplayLoader(request(false), 100, 100)(), { name: 'TimeoutError' });
});
