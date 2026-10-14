import type { DisplayData } from './display-data';
import type { Snapshot } from './types';

const MAX_BYTES = 32 * 1024 * 1024;

/** Expand the server-validated dictionary without copying shared metadata. */
export function expandDisplay(artifact: DisplayData): Snapshot {
  if (artifact?.version !== 1 || !artifact.payload || !Array.isArray(artifact.payload.results)) throw new Error('Invalid display data');
  const p = artifact.payload;
  const benchmarks = p.definitions.map(i => {
    const benchmark = p.benchmarks[i];
    if (!benchmark) throw new Error('Invalid display reference');
    return benchmark;
  });
  const results = p.results.map(r => {
    const model = p.models[r.model], benchmark = p.benchmarks[r.benchmark];
    if (!model || !benchmark) throw new Error('Invalid display reference');
    return { ...r, ...model, benchmark };
  });
  return { benchmarks, categories: p.categories, results, issues: [],
    sources: [{ name: artifact.source?.repo ?? 'local', files: results.length }],
    display: { generatedAt: artifact.generatedAt, source: artifact.source, digest: artifact.digest } };
}

/** One request per page lifetime; rejected requests are retryable. */
export function createDisplayLoader(request: typeof fetch = fetch): () => Promise<Snapshot> {
  let pending: Promise<Snapshot> | undefined;
  return () => pending ??= (async () => {
    const response = await request('/api/display/', { signal: AbortSignal.timeout(30_000) });
    if (!response.ok) { await response.body?.cancel(); throw new Error(`Display request failed: ${response.status}`); }
    if (Number(response.headers.get('content-length')) > MAX_BYTES) { await response.body?.cancel(); throw new Error('Display size limit exceeded'); }
    if (!response.body) throw new Error('Missing display data');
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let size = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > MAX_BYTES) throw new Error('Display size limit exceeded');
        chunks.push(value);
      }
    } finally { await reader.cancel(); }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    return expandDisplay(JSON.parse(new TextDecoder().decode(bytes)));
  })().catch(error => { pending = undefined; throw error; });
}

export const loadDisplay = createDisplayLoader();
