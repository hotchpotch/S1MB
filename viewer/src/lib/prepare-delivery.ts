/** Validate and compress public data before constructing the runtime image. */
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, rename, rm } from 'node:fs/promises';
import { gzip } from 'node:zlib';
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
  const compressed = await promisify(gzip)(body);
  const digest = createHash('sha256').update(body).digest('hex');
  const directory = `${file}.delivery`;
  await mkdir(directory, { recursive: true });
  // Content-addressed bytes are complete before the atomic manifest becomes visible.
  await atomicBytes(path.join(directory, `${digest}.json`), body);
  await atomicBytes(path.join(directory, `${digest}.gz`), compressed);
  await atomicJson(path.join(directory, 'manifest.json'), { version: 1, digest, bytes: body.length, gzipBytes: compressed.length });
  return { bytes: body.length, gzipBytes: compressed.length, results: artifact.payload.results.length };
}
