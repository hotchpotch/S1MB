/** Serve build-prepared bytes. No artifact validation, hashing or compression at runtime. */
import { open } from 'node:fs/promises';
import path from 'node:path';

async function boundedBytes(file: string, maximum: number) {
  const handle = await open(file, 'r');
  try {
    const size = (await handle.stat()).size;
    if (size > maximum) throw new Error('Prepared delivery exceeds its size limit');
    const bytes = Buffer.alloc(size);
    let offset = 0;
    while (offset < size) {
      const { bytesRead } = await handle.read(bytes, offset, size - offset, offset);
      if (!bytesRead) throw new Error('Incomplete prepared delivery');
      offset += bytesRead;
    }
    return bytes;
  } finally { await handle.close(); }
}
type Delivery = { body: string; gzip: Uint8Array<ArrayBuffer>; brotli: Uint8Array<ArrayBuffer>; etag: string };
export function createDisplayStore(file: string): () => Promise<Delivery> {
  let pending: Promise<Delivery> | undefined;
  return () => pending ??= (async () => {
    const directory = `${file}.delivery`;
    const manifest = JSON.parse((await boundedBytes(path.join(directory, 'manifest.json'), 1024)).toString());
    if (manifest.version !== 2 || typeof manifest.digest !== 'string' || !/^[a-f0-9]{64}$/.test(manifest.digest)) throw new Error('Invalid prepared delivery manifest');
    const body = await boundedBytes(path.join(directory, `${manifest.digest}.json`), 32 * 1024 * 1024);
    const gzip = await boundedBytes(path.join(directory, `${manifest.digest}.gz`), 33 * 1024 * 1024);
    const brotli = await boundedBytes(path.join(directory, `${manifest.digest}.br`), 33 * 1024 * 1024);
    if (body.length !== manifest.bytes || gzip.length !== manifest.gzipBytes || brotli.length !== manifest.brotliBytes) throw new Error('Incomplete prepared delivery');
    return { body: body.toString(), gzip: new Uint8Array(gzip), brotli: new Uint8Array(brotli), etag: `"${manifest.digest}"` };
  })().catch(error => { pending = undefined; throw error; });
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
    const accepted = new Map((request.headers.get('accept-encoding') ?? '').split(',').map(value => {
      const [encoding, ...parameters] = value.trim().split(';');
      const quality = parameters.find(p => /^\s*q\s*=/.test(p));
      return [encoding.trim().toLowerCase(), quality ? Number(quality.split('=')[1]) : 1];
    }));
    const quality = (encoding: string) => accepted.get(encoding) ?? accepted.get('*') ?? 0;
    const encoding = quality('br') > 0 && quality('br') >= quality('gzip') ? 'br' : quality('gzip') > 0 ? 'gzip' : undefined;
    const body = encoding === 'br' ? data.brotli : encoding === 'gzip' ? data.gzip : data.body;
    const etag = encoding ? `${data.etag.slice(0, -1)}-${encoding}"` : data.etag;
    const headers: Record<string, string> = { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'public, no-cache', ETag: etag, Vary: 'Accept-Encoding' };
    if (encoding) headers['Content-Encoding'] = encoding;
    headers['Content-Length'] = String(typeof body === 'string' ? Buffer.byteLength(body) : body.byteLength);
    const tags = request.headers.get('if-none-match')?.split(',').map(tag => tag.trim().replace(/^W\//, ''));
    if (tags?.some(tag => tag === '*' || tag === etag)) return new Response(null, { status: 304, headers });
    return new Response(body, { headers });
  } catch {
    return Response.json({ error: 'Display data is unavailable. Please retry.' }, { status: 503, headers: { 'Cache-Control': 'no-store' } });
  }
}
