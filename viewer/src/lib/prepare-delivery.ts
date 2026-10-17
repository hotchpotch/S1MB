/** Validate and compress public data before constructing the runtime image. */
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, rename, rm } from 'node:fs/promises';
import { gzip, brotliCompress, constants } from 'node:zlib';
import { promisify } from 'node:util';
import path from 'node:path';
import { readDisplay, atomicJson } from './display-data';

async function atomicBytes(file: string, bytes: Uint8Array) {
  const temporary = `${file}.${randomUUID()}.tmp`;
  try {
    const handle = await open(temporary, 'wx');
    try { await handle.writeFile(bytes); await handle.sync(); } finally { await handle.close(); }
    await rename(temporary, file);
  } finally { await rm(temporary, { force: true }); }
}

export async function prepareDelivery(file: string) {
  const { artifact } = await readDisplay(file);
  const body = Buffer.from(JSON.stringify(artifact));
  const [compressed, brotli] = await Promise.all([
    promisify(gzip)(body),
    promisify(brotliCompress)(body, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } }),
  ]);
  const digest = createHash('sha256').update(body).digest('hex');
  const directory = `${file}.delivery`;
  await mkdir(directory, { recursive: true });
  // Content-addressed bytes are complete before the atomic manifest becomes visible.
  await atomicBytes(path.join(directory, `${digest}.json`), body);
  await atomicBytes(path.join(directory, `${digest}.gz`), compressed);
  await atomicBytes(path.join(directory, `${digest}.br`), brotli);
  await atomicJson(path.join(directory, 'manifest.json'), { version: 2, digest, bytes: body.length, gzipBytes: compressed.length, brotliBytes: brotli.length });
  return { bytes: body.length, gzipBytes: compressed.length, brotliBytes: brotli.length, results: artifact.payload.results.length };
}
