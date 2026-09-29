/** Read saved, publication-validated measurements. Never load evaluation inputs. */
import { readdir, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { z } from 'zod';
import { metadataSchema, readResultJson } from './result-files';
import { METRICS, type Benchmark, type Category, type ResultSummary, type Snapshot } from './types';

const id = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/);
const count = z.number().int().nonnegative();
const benchmarkSchema = z.object({ id, task: z.enum(['choice', 'noul', 'score']), dataset: z.string(), split: z.string(), case_count: count.positive(), decision_count: count.positive(), primary_metric: z.string() });
const categorySchema = z.object({ id, name: z.string(), description: z.string(), benchmarks: z.array(id).nonempty() });
const resultSchema = z.object({
  format_version: z.literal(1), run_id: id, benchmark: benchmarkSchema,
  model: z.object({ id: z.string(), adapter: z.string(), settings: z.object({ questions_per_call: count.positive().nullish() }),
    total_params: count.nullish(), active_params: count.nullish(), parameter_count_method: z.literal('non_lookup_parameters_v1').nullish() }).refine(m => m.active_params == null || (m.parameter_count_method === 'non_lookup_parameters_v1' && (m.total_params == null || m.active_params <= m.total_params)), 'Invalid parameter counts'),
  provenance: z.enum(['measured', 'demo']), status: z.enum(['complete', 'partial']),
  counts: z.object({ cases: count, expected: count, succeeded: count, failed: count }),
  metrics: z.record(z.string(), z.number().finite().nullable()),
});
type Metadata = z.infer<typeof metadataSchema>;
type Entry = { file: string; stamp: string; kind: 'benchmark' | 'category' | 'metadata' | 'result'; published: boolean };
type Parsed = Benchmark | Category | Metadata | ResultSummary;
const epsilon = 1e-12;

export function summarize(value: unknown): ResultSummary {
  const r = resultSchema.parse(value);
  const b = r.benchmark, c = r.counts, m = r.metrics;
  if (b.primary_metric !== METRICS[b.task].name) throw new Error('Unknown primary metric');
  const bounded = (name: string): number | null => {
    const n = m[name];
    if (n === undefined || (n !== null && (n < 0 || n > 1))) throw new Error(`Missing or invalid metric: ${name}`);
    return n;
  };
  const primary = bounded(b.primary_metric);
  const adjusted = bounded('baseline_adjusted_score');
  if (c.expected !== b.decision_count || c.cases > b.case_count || c.succeeded + c.failed > c.expected ||
      (c.succeeded === 0) !== (primary === null)) throw new Error('Inconsistent result counts');
  const complete = c.failed === 0 && c.succeeded === c.expected && c.cases === b.case_count;
  if ((r.status === 'complete') !== complete) throw new Error('Incorrect completion status');
  if (r.model.adapter === 'dummy' && r.provenance !== 'demo') throw new Error('Dummy results must be demo');
  let eligible: boolean, reason: string, skill: number | null;
  if (b.task === 'choice') {
    const baseline = bounded('fixed_answer_accuracy_baseline');
    if (baseline === null) throw new Error('Missing Choice baseline');
    eligible = baseline < 1 - epsilon;
    reason = 'A fixed answer already achieves 100%: no headroom for adjustment.';
    skill = eligible && primary !== null ? (primary - baseline) / (1 - baseline) : null;
  } else if (b.task === 'score') {
    const baseline = bounded('constant_mae_baseline');
    if (baseline === null) throw new Error('Missing Score baseline');
    eligible = baseline > epsilon;
    reason = 'Constant targets: baseline MAE is zero.';
    skill = eligible && primary !== null ? 1 - primary / baseline : null;
  } else {
    const prevalence = bounded('positive_prevalence'), balanced = bounded('balanced_accuracy');
    if (prevalence === null) throw new Error('Missing Noul prevalence');
    eligible = prevalence > epsilon && prevalence < 1 - epsilon;
    reason = 'Only one target class: balanced accuracy is undefined.';
    skill = balanced === null ? null : 2 * balanced - 1;
  }
  const expected = skill === null ? null : Math.max(0, Math.min(1, skill));
  if (adjusted === null || expected === null ? adjusted !== expected : Math.abs(adjusted - expected) > 1e-8) throw new Error('Inconsistent adjusted score');
  const { format_version: _format, ...summary } = r;
  return { ...summary, scoring: { eligible, reason: eligible ? null : reason } };
}

async function boundedMap<T, R>(values: T[], fn: (value: T) => Promise<R>, width = 8): Promise<R[]> {
  const out: R[] = [];
  for (let i = 0; i < values.length; i += width) out.push(...await Promise.all(values.slice(i, i + width).map(fn)));
  return out;
}

/** Only filesystem metadata is read on unchanged checks; symlinked roots are supported. */
async function scan(dataDir: string, resultDirs: string[]): Promise<Entry[]> {
  const entries: Entry[] = [];
  async function add(file: string, kind: Entry['kind'], published = false) {
    const s = await stat(file, { bigint: true });
    if (!s.isFile() || s.size > 64n * 1024n * 1024n) throw new Error('Invalid or oversized result file');
    entries.push({ file, kind, published, stamp: `${s.size}:${s.mtimeNs}:${s.ctimeNs}:${s.ino}` });
    if (entries.length > 20_000) throw new Error('Too many result files');
  }
  for (const [dir, kind] of [['benchmarks', 'benchmark'], ['categories', 'category']] as const) {
    const base = path.join(dataDir, dir);
    await boundedMap(await readdir(base), async name => {
      if (name.endsWith('.json')) await add(path.join(base, name), kind);
    });
  }
  async function walk(directory: string, depth = 0) {
    if (depth > 4) throw new Error('Result directory nesting exceeds limit');
    const children = await readdir(directory, { withFileTypes: true });
    const published = children.some(e => e.isFile() && e.name === 'metadata.json');
    await boundedMap(children.filter(e => !e.name.startsWith('.')), async e => {
      const file = path.join(directory, e.name);
      if (e.isDirectory()) await walk(file, depth + 1);
      else if (e.isFile()) {
        if (e.name === 'metadata.json') await add(file, 'metadata', true);
        else if (e.name.endsWith('.json.xz') || (!published && e.name.endsWith('.json'))) {
          if (e.name.endsWith('.xz') && !published) throw new Error('Published result requires metadata.json');
          await add(file, 'result', published);
        }
      }
    });
  }
  for (const directory of [...new Set(await Promise.all(resultDirs.map(d => realpath(d))))]) await walk(directory);
  return entries.sort((a, b) => a.file.localeCompare(b.file));
}
function signature(entries: Entry[]): string { return JSON.stringify(entries); }

/** Per-file summaries stay in the worker, so unchanged XZ files are never decoded again. */
export class FilesystemResults {
  private previous = new Map<string, { stamp: string; value: Parsed }>();
  private fingerprint = '';
  constructor(private dataDir: string, private resultDirs: string[], private read = readResultJson) {}
  async refresh(): Promise<Snapshot | null> {
    const entries = await scan(this.dataDir, this.resultDirs);
    const fingerprint = signature(entries);
    if (fingerprint === this.fingerprint) return null;
    const next = new Map<string, { stamp: string; value: Parsed }>();
    await boundedMap(entries, async e => {
      const old = this.previous.get(e.file);
      if (old?.stamp === e.stamp) { next.set(e.file, old); return; }
      try {
        const raw = await this.read(e.file);
        const value = e.kind === 'benchmark' ? benchmarkSchema.parse(raw) : e.kind === 'category' ? categorySchema.parse(raw)
          : e.kind === 'metadata' ? metadataSchema.parse(raw) : summarize(raw);
        next.set(e.file, { stamp: e.stamp, value });
      } catch { throw new Error(`Invalid ${e.kind} file: ${path.basename(e.file)}`); }
    }, 4);
    const definitions = entries.filter(e => e.kind === 'benchmark').map(e => next.get(e.file)!.value as Benchmark);
    const categories = entries.filter(e => e.kind === 'category').map(e => next.get(e.file)!.value as Category);
    const known = new Map(definitions.map(b => [b.id, b]));
    if (!definitions.length || !categories.length || known.size !== definitions.length || new Set(categories.map(c => c.id)).size !== categories.length) throw new Error('Missing or duplicate definitions');
    for (const b of definitions) if (b.primary_metric !== METRICS[b.task].name) throw new Error('Unknown primary metric');
    for (const c of categories) if (new Set(c.benchmarks).size !== c.benchmarks.length || c.benchmarks.some(id => !known.has(id))) throw new Error('Invalid category membership');
    const metadata = new Map<string, Metadata>();
    for (const e of entries.filter(e => e.kind === 'metadata')) {
      const m = next.get(e.file)!.value as Metadata;
      if (m.model_id !== path.basename(path.dirname(e.file))) throw new Error('Model directory/metadata ID mismatch');
      metadata.set(path.dirname(e.file), m);
    }
    const results = new Map<string, ResultSummary>();
    const identities = new Map<string, string>();
    for (const e of entries.filter(e => e.kind === 'result')) {
      const original = next.get(e.file)!.value as ResultSummary;
      const current = known.get(original.benchmark.id);
      if (!current) continue; // Current definitions control leaderboard membership.
      for (const key of ['id', 'task', 'dataset', 'split', 'primary_metric'] as const) {
        if (current[key] !== original.benchmark[key]) throw new Error('Incompatible benchmark definition');
      }
      const m = metadata.get(path.dirname(e.file));
      if (e.published && (!m || path.basename(e.file) !== `${original.benchmark.id}.json.xz`)) throw new Error('Invalid published result filename');
      const metadataCounts = m && (m.total_params != null || m.active_params != null);
      const model = m ? { ...original.model, display_name: m.display_name, short_name: m.short_name, url: m.url, hf_url: m.hf_url,
        total_params: metadataCounts ? m.total_params : original.model.total_params,
        active_params: metadataCounts ? m.active_params : original.model.active_params,
        parameter_count_method: metadataCounts ? m.parameter_count_method : original.model.parameter_count_method } : original.model;
      const r = { ...original, run_id: m?.model_id ?? original.run_id, model };
      const identity = JSON.stringify(m ?? model);
      if (identities.has(r.run_id) && identities.get(r.run_id) !== identity) throw new Error('Conflicting model identity');
      identities.set(r.run_id, identity);
      const key = JSON.stringify([r.run_id, r.benchmark.id]);
      if (results.has(key) && JSON.stringify(results.get(key)) !== JSON.stringify(r)) throw new Error('Conflicting result');
      results.set(key, r);
    }
    // Never publish a mixture if files changed while the candidate was being loaded.
    if (signature(await scan(this.dataDir, this.resultDirs)) !== fingerprint) throw new Error('Results changed during refresh; keeping previous snapshot');
    this.previous = next;
    this.fingerprint = fingerprint;
    return { categories, benchmarks: definitions, results: [...results.values()], issues: [], sources: this.resultDirs.map(d => ({ name: path.basename(d), files: entries.filter(e => e.kind === 'result' && e.file.startsWith(path.resolve(d) + path.sep)).length })) };
  }
}
