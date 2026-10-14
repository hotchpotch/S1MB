/** Serve one validated, immutable local artifact. No remote I/O or source scans. */
import { readDisplay } from './display-data';
import { createHash } from 'node:crypto';
import { gzip } from 'node:zlib';
import { promisify } from 'node:util';

const compress = promisify(gzip);
type Delivery = { body: string; gzip: Uint8Array<ArrayBuffer>; etag: string };
export function createDisplayStore(file: string): () => Promise<Delivery> {
  let pending: Promise<Delivery> | undefined;
  return () => pending ??= readDisplay(file).then(async ({ artifact }) => {
    const body = JSON.stringify(artifact);
    return { body, gzip: new Uint8Array(await compress(body)), etag: `"${createHash('sha256').update(body).digest('hex')}"` };
  }).catch(error => { pending = undefined; throw error; });
}

const shared = globalThis as typeof globalThis & { s1mbDelivery?: () => Promise<Delivery> };
function getDelivery() {
  const file = process.env.S1MB_DISPLAY_FILE;
  if (!file) throw new Error('Start with npm start -- --display-file /path/to/viewer-summary.json');
  shared.s1mbDelivery ??= createDisplayStore(file);
  return shared.s1mbDelivery().catch(error => {
    console.error('Unable to load bundled display data', error);
    throw error;
  });
}

export async function displayResponse(request: Request, load = getDelivery): Promise<Response> {
  try {
    const data = await load();
    const compressed = request.headers.get('accept-encoding')?.split(',').some(value => {
      const [encoding, ...parameters] = value.trim().split(';');
      return encoding.trim() === 'gzip' && !parameters.some(p => /^\s*q\s*=/.test(p) && Number(p.split('=')[1]) === 0);
    });
    const etag = compressed ? `${data.etag.slice(0, -1)}-gzip"` : data.etag;
    const headers: Record<string, string> = { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'public, no-cache', ETag: etag, Vary: 'Accept-Encoding' };
    if (compressed) headers['Content-Encoding'] = 'gzip';
    const tags = request.headers.get('if-none-match')?.split(',').map(tag => tag.trim().replace(/^W\//, ''));
    if (tags?.some(tag => tag === '*' || tag === etag)) return new Response(null, { status: 304, headers });
    return new Response(compressed ? data.gzip : data.body, { headers });
  } catch {
    return Response.json({ error: 'Display data is unavailable. Please retry.' }, { status: 503, headers: { 'Cache-Control': 'no-store' } });
  }
}
