/** Immutable, checksummed generations. One viewer owns each cache namespace. */
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readdir, readFile, stat, unlink } from 'node:fs/promises';
import path from 'node:path';
import { z } from 'zod';
import { benchmarkSchema, categorySchema, resultSchema, summarize } from './result-loader';
import type { Snapshot } from './types';

// Bump when summary semantics or the persisted representation changes incompatibly.
const version = 1;
const limit = 64 * 1024 * 1024;
const generationName = /^\d{13}-[a-f0-9-]{36}\.json$/;
const sha = (value: string) => createHash('sha256').update(value).digest('hex');
const link = z.url().refine(value => ['http:', 'https:'].includes(new URL(value).protocol)).nullish();
const model = resultSchema.shape.model.safeExtend({ display_name: z.string().optional(), short_name: z.string().optional(), url: link, hf_url: link });
const summary = resultSchema.omit({ format_version: true }).extend({ model,
  scoring: z.object({ eligible: z.boolean(), reason: z.string().nullable() }),
});
const snapshotSchema = z.object({ benchmarks: z.array(benchmarkSchema), categories: z.array(categorySchema),
  results: z.array(summary), issues: z.array(z.string()), sources: z.array(z.object({ name: z.string(), files: z.number().int().nonnegative() })),
}).strict();
const envelopeSchema = z.object({ version: z.literal(version), source: z.string(), savedAt: z.iso.datetime(), digest: z.string(), snapshot: z.unknown() }).strict();

function checkedSnapshot(value: unknown): Snapshot {
  const s = snapshotSchema.parse(value);
  const ids = new Set(s.benchmarks.map(b => b.id));
  if (!ids.size || ids.size !== s.benchmarks.length || !s.categories.length || s.categories.some(c => c.benchmarks.some(id => !ids.has(id)))) throw new Error('Invalid cached definitions');
  const keys = new Set<string>();
  for (const r of s.results) {
    const checked = summarize({ ...r, format_version: 1 });
    const key = JSON.stringify([r.run_id, r.benchmark.id]);
    if (!ids.has(r.benchmark.id) || keys.has(key) || JSON.stringify(checked.scoring) !== JSON.stringify(r.scoring)) throw new Error('Invalid cached result');
    keys.add(key);
  }
  return s;
}

type Generation = { name: string; digest: string; savedAt: string; snapshot: Snapshot };
export class SnapshotStore {
  readonly directory: string;
  private source: string;
  constructor(directory: string, dataDir: string, dirs: string[]) {
    // Use configured paths, never resolved inode numbers or mount-specific realpaths.
    this.source = sha(JSON.stringify({ version, data: path.resolve(dataDir), results: dirs.map(d => path.resolve(d)) }));
    this.directory = path.join(directory, `v${version}-${this.source}`);
  }
  private async names(): Promise<string[]> {
    try { return (await readdir(this.directory)).filter(n => generationName.test(n)).sort().reverse(); }
    catch (e) { if ((e as NodeJS.ErrnoException).code === 'ENOENT') return []; throw e; }
  }
  private async read(name: string): Promise<Generation> {
    const file = path.join(this.directory, name);
    if ((await stat(file)).size > limit) throw new Error('Cached snapshot exceeds size limit');
    const raw = envelopeSchema.parse(JSON.parse(await readFile(file, 'utf8')));
    if (raw.source !== this.source || sha(JSON.stringify(raw.snapshot)) !== raw.digest) throw new Error('Cache checksum or source mismatch');
    return { name, digest: raw.digest, savedAt: raw.savedAt, snapshot: checkedSnapshot(raw.snapshot) };
  }
  async restore(): Promise<Snapshot | null> {
    // Stop at the first valid generation: startup does not inspect the result tree.
    for (const name of await this.names()) {
      try {
        const g = await this.read(name);
        console.log(`Restored results cache (${g.snapshot.results.length} results).`);
        return { ...g.snapshot, cache: { checkedAt: g.savedAt, refreshFailed: false } };
      } catch { console.warn('Ignoring invalid results cache generation.'); }
    }
    return null;
  }
  async save(snapshot: Snapshot): Promise<void> {
    const { cache: _cache, ...data } = snapshot;
    const clean = checkedSnapshot(data);
    const digest = sha(JSON.stringify(clean));
    await mkdir(this.directory, { recursive: true });
    const names = await this.names();
    const valid: Generation[] = [];
    for (const name of names) {
      try { valid.push(await this.read(name)); } catch { /* Removed only after a good generation is available. */ }
    }
    if (valid[0]?.digest !== digest) {
      const timestamp = Math.max(Date.now(), ...names.map(n => Number(n.slice(0, 13)) + 1));
      const name = `${timestamp}-${randomUUID()}.json`;
      const payload = JSON.stringify({ version, source: this.source, savedAt: new Date().toISOString(), digest, snapshot: clean });
      if (Buffer.byteLength(payload) > limit) throw new Error('Cached snapshot exceeds size limit');
      try {
        const file = await open(path.join(this.directory, name), 'wx', 0o600);
        try {
          await file.writeFile(payload);
          // Local durability where supported. Managed mounts define their own flush semantics.
          try { await file.sync(); } catch (e) {
            if (!['EINVAL', 'ENOSYS', 'ENOTSUP'].includes((e as NodeJS.ErrnoException).code ?? '')) throw e;
          }
        } finally { await file.close(); }
        const written = await this.read(name);
        if (written.digest !== digest) throw new Error('Cache read-back verification failed');
        valid.unshift(written);
      } catch (error) {
        // Repeated failed saves must not accumulate partial generations.
        await unlink(path.join(this.directory, name)).catch(() => {});
        throw error;
      }
      console.log(`Saved results cache (${clean.results.length} results).`);
    }
    // Never delete the fallback until the new complete JSON has been read and verified.
    const keep = new Set(valid.slice(0, 2).map(g => g.name));
    for (const name of names) if (!keep.has(name)) await unlink(path.join(this.directory, name));
  }
}
