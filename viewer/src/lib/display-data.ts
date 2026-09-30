import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, rename, stat, unlink } from 'node:fs/promises';
import path from 'node:path';
import { z } from 'zod';
import { benchmarkSchema, categorySchema, resultSchema, summarize } from './result-loader';
import type { Snapshot } from './types';

const link = z.url().refine(value => ['http:', 'https:'].includes(new URL(value).protocol)).nullish();
const model = resultSchema.shape.model.safeExtend({ display_name: z.string().optional(), short_name: z.string().optional(), url: link, hf_url: link });
const summary = resultSchema.omit({ format_version: true }).extend({ model,
  scoring: z.object({ eligible: z.boolean(), reason: z.string().nullable() }),
});
const snapshotSchema = z.object({ benchmarks: z.array(benchmarkSchema), categories: z.array(categorySchema),
  results: z.array(summary), issues: z.array(z.string()), sources: z.array(z.object({ name: z.string(), files: z.number().int().nonnegative() })),
}).strict();
export function checkedSnapshot(value: unknown): Snapshot {
  const s = snapshotSchema.parse(value);
  const ids = new Set(s.benchmarks.map(b => b.id));
  if (!ids.size || ids.size !== s.benchmarks.length || !s.categories.length || s.categories.some(c => c.benchmarks.some(id => !ids.has(id)))) throw new Error('Invalid display definitions');
  if (s.issues.length || new Set(s.categories.map(c => c.id)).size !== s.categories.length || s.categories.some(c => new Set(c.benchmarks).size !== c.benchmarks.length)) throw new Error('Invalid display categories or issues');
  const keys = new Set<string>();
  for (const r of s.results) {
    const current = s.benchmarks.find(b => b.id === r.benchmark.id);
    if (!current || ['task', 'dataset', 'split', 'primary_metric'].some(k => current[k as keyof typeof current] !== r.benchmark[k as keyof typeof r.benchmark])) throw new Error('Incompatible display benchmark');
    const checked = summarize({ ...r, format_version: 1 });
    const key = JSON.stringify([r.run_id, r.benchmark.id]);
    if (!ids.has(r.benchmark.id) || keys.has(key) || JSON.stringify(checked.scoring) !== JSON.stringify(r.scoring)) throw new Error('Invalid display result');
    keys.add(key);
  }
  return s;
}


export const SUMMARY_FILE = 'viewer-summary.json';
export const MAX_BYTES = 32 * 1024 * 1024;
const sha = (text: string) => createHash('sha256').update(text).digest('hex');
const index = z.number().int().nonnegative();
const sourceSchema = z.object({ repo: z.string().regex(/^[\w.-]+\/[\w.-]+$/), revision: z.string().regex(/^[a-f0-9]{40}$/) }).strict();
const payloadSchema = z.object({
  benchmarks: z.array(benchmarkSchema), definitions: z.array(index), categories: z.array(categorySchema),
  models: z.array(z.object({ run_id: z.string(), model }).strict()),
  results: z.array(summary.omit({ run_id: true, model: true, benchmark: true }).extend({ model: index, benchmark: index }).strict()),
}).strict();
const artifactSchema = z.object({
  version: z.literal(1), generatedAt: z.iso.datetime(), source: sourceSchema.nullable(),
  digest: z.string().regex(/^[a-f0-9]{64}$/), payload: payloadSchema,
}).strict();
export type DisplayData = z.infer<typeof artifactSchema>;

export function decodeDisplay(value: unknown): { artifact: DisplayData; snapshot: Snapshot } {
  const a = artifactSchema.parse(value), p = a.payload;
  if (sha(JSON.stringify(p)) !== a.digest) throw new Error('Display JSON checksum mismatch');
  const results = p.results.map(r => {
    const m = p.models[r.model], b = p.benchmarks[r.benchmark];
    if (!m || !b) throw new Error('Invalid display reference');
    return { ...r, ...m, benchmark: b };
  });
  const snapshot = checkedSnapshot({ benchmarks: p.definitions.map(i => p.benchmarks[i]), categories: p.categories,
    results, issues: [], sources: [{ name: a.source?.repo ?? 'local', files: results.length }] });
  if (!results.length || new Set(results.map(r => r.run_id)).size !== p.models.length) throw new Error('Empty display data or unused model');
  return { artifact: a, snapshot: { ...snapshot, display: { generatedAt: a.generatedAt, source: a.source, digest: a.digest } } };
}

export function encodeDisplay(value: Snapshot, source: DisplayData['source']): DisplayData {
  const { display: _display, ...data } = value;
  const snapshot = checkedSnapshot(data);
  const benchmarks: DisplayData['payload']['benchmarks'] = [], models: DisplayData['payload']['models'] = [];
  const benchmarkIds = new Map<string, number>(), modelIds = new Map<string, number>();
  const benchmarkIndex = (b: Snapshot['benchmarks'][number]) => {
    const key = JSON.stringify(b);
    if (!benchmarkIds.has(key)) { benchmarkIds.set(key, benchmarks.length); benchmarks.push(b); }
    return benchmarkIds.get(key)!;
  };
  const definitions = snapshot.benchmarks.map(benchmarkIndex);
  const results = snapshot.results.map(({ run_id, model, benchmark, ...r }) => {
    if (!modelIds.has(run_id)) { modelIds.set(run_id, models.length); models.push({ run_id, model }); }
    else if (JSON.stringify(models[modelIds.get(run_id)!].model) !== JSON.stringify(model)) throw new Error('Conflicting display model');
    return { ...r, model: modelIds.get(run_id)!, benchmark: benchmarkIndex(benchmark) };
  });
  const payload = { benchmarks, definitions, categories: snapshot.categories, models, results };
  const artifact = { version: 1 as const, generatedAt: new Date().toISOString(), source, digest: sha(JSON.stringify(payload)), payload };
  return decodeDisplay(artifact).artifact;
}

export async function readDisplay(file: string) {
  if ((await stat(file)).size > MAX_BYTES) throw new Error('Display JSON exceeds size limit');
  return decodeDisplay(JSON.parse(await readFile(file, 'utf8')));
}

export async function atomicJson(file: string, value: unknown) {
  const content = JSON.stringify(value);
  if (Buffer.byteLength(content) > MAX_BYTES) throw new Error('Display JSON exceeds size limit');
  await mkdir(path.dirname(file), { recursive: true });
  const temp = `${file}.${randomUUID()}.tmp`;
  try {
    const handle = await open(temp, 'wx', 0o600);
    try { await handle.writeFile(content); await handle.sync(); } finally { await handle.close(); }
    await rename(temp, file);
  } finally { await unlink(temp).catch(() => {}); }
}

/** Any lost measurement or completion requires a human, even if totals increased. */
export function reductionWarnings(before: Snapshot, after: Snapshot): string[] {
  const warnings: string[] = [];
  const next = new Map(after.results.map(r => [JSON.stringify([r.run_id, r.benchmark.id]), r]));
  for (const r of before.results) {
    const current = next.get(JSON.stringify([r.run_id, r.benchmark.id]));
    if (!current) warnings.push(`Removed result: ${r.run_id} / ${r.benchmark.id}`);
    else if (r.status === 'complete' && current.status !== 'complete') warnings.push(`Completion regressed: ${r.run_id} / ${r.benchmark.id}`);
    else if (current.counts.succeeded < r.counts.succeeded) warnings.push(`Fewer successful decisions: ${r.run_id} / ${r.benchmark.id}`);
  }
  for (const b of before.benchmarks) if (!after.benchmarks.some(n => n.id === b.id)) warnings.push(`Removed definition: ${b.id}`);
  for (const c of before.categories) {
    const current = after.categories.find(n => n.id === c.id);
    if (!current) warnings.push(`Removed category: ${c.id}`);
    else for (const id of c.benchmarks) if (!current.benchmarks.includes(id)) warnings.push(`Removed category member: ${c.id} / ${id}`);
  }
  const oldBytes = Buffer.byteLength(JSON.stringify(before.results)), newBytes = Buffer.byteLength(JSON.stringify(after.results));
  if (newBytes < oldBytes * 0.8) warnings.push(`Summary content shrank by more than 20%: ${oldBytes} -> ${newBytes} bytes`);
  return warnings;
}
