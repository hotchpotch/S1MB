/** Read local files once per process; components consume only summaries. */
import { readdir, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { readResultJson, metadataSchema, datasetSourceSchema } from './result-files';
import { z } from 'zod';
import { METRICS, type Snapshot } from './types';
import { prepareScoring } from './diagnostics';
import { loadHfCases } from './hf-data';
import { HubResultsCache, hubResultsOptions } from './hub-results';

const id = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/);
const count = z.number().int().nonnegative();
const benchmark = z.object({ id, task: z.enum(['choice', 'noul', 'score']), dataset: z.string(), split: z.literal('test'), case_count: count.positive(), decision_count: count.positive(), primary_metric: z.string() }).strict();
const category = z.object({ id, name: z.string(), description: z.string(), benchmarks: z.array(id).nonempty() }).strict();
const prediction = z.object({ case_id: z.string(), question_id: z.string(), probabilities: z.record(z.string(), z.number().min(0).max(1)).nullable(), error: z.string().nullable() }).strict();
const resultSchema = z.object({
  format_version: z.literal(1), run_id: id, benchmark,
  model: z.object({ id: z.string(), adapter: z.string(), revision: z.string(), settings: z.record(z.string(), z.unknown()), total_params: count.nullish(), active_params: count.nullish(), parameter_count_method: z.literal('non_lookup_parameters_v1').nullish() }).strict().refine(m => (m.active_params == null || (m.total_params != null && m.active_params <= m.total_params)) && (m.active_params != null) === (m.parameter_count_method != null), 'Invalid parameter counts'),
  created_at: z.string().datetime({ offset: true }), evaluator_version: z.string(),
  provenance: z.enum(['measured', 'demo']), status: z.enum(['complete', 'partial']),
  counts: z.object({ cases: count, expected: count, succeeded: count, failed: count }).strict(),
  metrics: z.record(z.string(), z.number().nullable()), elapsed_seconds: z.number().nonnegative(),
  environment: z.record(z.string(), z.unknown()), predictions: z.array(prediction),
}).strict();

export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value !== null && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`).join(',')}}`;
  return JSON.stringify(value);
}

const json = readResultJson;
async function files(root: string): Promise<string[]> {
  const output: string[] = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const file = path.join(root, entry.name);
    // Directory links inside an external result root are not followed.
    if (entry.isDirectory()) output.push(...await files(file));
    else if (entry.isFile() && (entry.name.endsWith('.json') || entry.name.endsWith('.json.xz')) && entry.name !== 'metadata.json') output.push(file);
  }
  return output.sort();
}

export async function loadSnapshot(dataDir: string, resultDirs: string[]): Promise<Snapshot> {
  const definitions = await Promise.all((await files(path.join(dataDir, 'benchmarks'))).map(async f => benchmark.parse(await json(f))));
  const categories = await Promise.all((await files(path.join(dataDir, 'categories'))).map(async f => category.parse(await json(f))));
  const known = new Map(definitions.map(b => [b.id, b]));
  if (known.size !== definitions.length || new Set(categories.map(c => c.id)).size !== categories.length) throw new Error('Duplicate definition IDs');
  for (const b of definitions) if (b.primary_metric !== METRICS[b.task].name) throw new Error(`Unknown metric: ${b.id}`);
  for (const c of categories) if (new Set(c.benchmarks).size !== c.benchmarks.length || c.benchmarks.some(b => !known.has(b))) throw new Error(`Invalid category: ${c.id}`);
  const snapshot: Snapshot = { categories, benchmarks: definitions, results: [], issues: [], sources: [] };
  const scoring = new Map<string, ReturnType<typeof prepareScoring>>();
  snapshot.scoring = {};
  const datasets = new Map<string, Awaited<ReturnType<typeof loadHfCases>>>();
  for (const b of definitions) {
    const key = canonical([b.dataset, b.split]);
    if (!datasets.has(key)) datasets.set(key, await loadHfCases(dataDir, b.dataset, b.split));
    const cases = datasets.get(key)!;
    const prepared = prepareScoring(cases, b.task);
    if (prepared.decisions !== b.decision_count || prepared.cases !== b.case_count) throw new Error(`Dataset counts differ: ${b.id}`);
    scoring.set(b.id, prepared);
    snapshot.scoring[b.id] = prepared.context;
  }
  const results = new Map<string, z.infer<typeof resultSchema>>();
  const presentation = new Map<string, { metadata: z.infer<typeof metadataSchema>; original_run_id: string }>();
  const contexts = new Map<string, ReturnType<typeof prepareScoring>['context']>();
  const computed = new Map<string, Record<string, number | null>>();
  const publishedRows = new Set<string>();
  const rowMetadata = new Map<string, string>();
  const inconsistentRows = new Set<string>();
  const installedSource = await json(path.join(dataDir, 'datasets/hub-source.json')).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error; });
  const metadataCache = new Map<string, z.infer<typeof metadataSchema> | null>();
  async function metadataFor(file: string) {
    const folder = path.dirname(file);
    if (!metadataCache.has(folder)) {
      const metadataFile = path.join(folder, 'metadata.json');
      const exists = await stat(metadataFile).then(() => true, (error: NodeJS.ErrnoException) => {
        if (error.code === 'ENOENT') return false;
        throw error;
      });
      const metadata = exists ? metadataSchema.parse(await json(metadataFile)) : null;
      if (metadata && metadata.model_id !== path.basename(folder)) throw new Error('Model directory/metadata ID mismatch');
      metadataCache.set(folder, metadata);
    }
    return metadataCache.get(folder)!;
  }
  const conflictKeys = new Set<string>();
  let validDirs = 0;
  for (const directory of resultDirs) {
    let entries: string[];
    try {
      const resolved = await realpath(directory);
      if (!(await stat(resolved)).isDirectory()) throw new Error('Not a directory');
      entries = await files(resolved);
      validDirs++;
    } catch {
      snapshot.issues.push(`Cannot read result directory: ${directory}`);
      continue;
    }
    snapshot.sources.push({ name: path.basename(directory), files: entries.length });
    for (const file of entries) {
      try {
        const r = resultSchema.parse(await json(file));
        if (r.model.adapter === 'dummy' && r.provenance !== 'demo') throw new Error('Dummy results must be marked as demo');
        const metadata = await metadataFor(file);
        const current = known.get(r.benchmark.id);
        const b = r.benchmark;
        if (!current) throw new Error('Unknown benchmark definition');
        let cases;
        let prepared;
        if (metadata) {
          if (path.basename(file) !== `${b.id}.json.xz`) throw new Error('Benchmark filename/ID mismatch');
          for (const key of ['id', 'task', 'dataset', 'split', 'primary_metric'] as const) {
            if (current[key] !== b[key]) throw new Error('Incompatible benchmark definition');
          }
          const source = datasetSourceSchema.parse(r.environment.dataset_source);
          const root = canonical(installedSource && typeof installedSource === 'object'
            ? { repo_id: (installedSource as Record<string, unknown>).repo_id, revision: (installedSource as Record<string, unknown>).revision } : null) === canonical(source)
            ? dataDir : path.join(dataDir, 'result-datasets', createHash('sha256').update(`${source.repo_id}@${source.revision}`).digest('hex'));
          const key = canonical([root, b.dataset, b.split]);
          if (!datasets.has(key)) datasets.set(key, await loadHfCases(root, b.dataset, b.split));
          cases = datasets.get(key)!;
          prepared = prepareScoring(cases, b.task);
          if (prepared.decisions !== b.decision_count || prepared.cases !== b.case_count) throw new Error('Dataset counts differ');
        } else {
          if (file.endsWith('.json.xz')) throw new Error('Published result requires metadata.json');
          if (canonical(current) !== canonical(b)) throw new Error('Changed benchmark definition');
          cases = datasets.get(canonical([b.dataset, b.split]))!;
          prepared = scoring.get(b.id)!;
        }
        const hashes = Object.fromEntries(cases.filter(c => c.questions.some(q => q.task === b.task)).map(c => [c.case_id, c.input_hash]));
        if (canonical(r.environment.input_hashes) !== canonical(hashes)) throw new Error('Result input hashes differ from the dataset or are missing');
        const recomputed = prepared.calculate(r.predictions);
        const storedKeys = Object.keys(r.metrics);
        if (!(b.primary_metric in r.metrics) || storedKeys.length !== Object.keys(recomputed).length) throw new Error('Unexpected metric');
        for (const [name, value] of Object.entries(r.metrics)) {
          const expected = recomputed[name];
          if (expected === undefined || (value === null || expected === null ? value !== expected : Math.abs(value - expected) > 1e-8)) throw new Error(`Stored metric differs from predictions: ${name}`);
        }
        const score = r.metrics[b.primary_metric];
        if (score !== null && (score < 0 || score > 1)) throw new Error('Metric outside [0, 1]');
        const seen = new Set<string>();
        let failed = 0;
        for (const p of r.predictions) {
          const key = canonical([p.case_id, p.question_id]);
          if (seen.has(key)) throw new Error('Duplicate prediction IDs');
          seen.add(key);
          if ((p.error === null) === (p.probabilities === null)) throw new Error('Missing or ambiguous prediction');
          if (p.error !== null) failed++;
          if (p.probabilities && Math.abs(Object.values(p.probabilities).reduce((a, b) => a + b, 0) - 1) > 1e-5) throw new Error('Probabilities do not sum to one');
        }
        if (r.counts.expected !== b.decision_count || r.counts.failed !== failed || r.counts.succeeded !== r.predictions.length - failed || r.counts.cases !== new Set(r.predictions.map(p => p.case_id)).size || r.counts.cases > b.case_count || r.predictions.length > b.decision_count) throw new Error('Prediction counts differ');
        const complete = failed === 0 && r.predictions.length === b.decision_count && r.counts.cases === b.case_count;
        if ((r.status === 'complete') !== complete || (r.counts.succeeded === 0) !== (score === null)) throw new Error('Incorrect completion or metric status');
        const original_run_id = r.run_id;
        if (metadata) r.run_id = metadata.model_id;
        const rowIdentity = canonical(metadata);
        if (rowMetadata.has(r.run_id) && rowMetadata.get(r.run_id) !== rowIdentity) inconsistentRows.add(r.run_id);
        rowMetadata.set(r.run_id, rowIdentity);
        const key = canonical([r.run_id, b.id]);
        computed.set(key, recomputed);
        contexts.set(key, prepared.context);
        if (metadata) {
          presentation.set(key, { metadata, original_run_id });
          publishedRows.add(r.run_id);
        }
        const prior = results.get(key);
        if (prior && canonical(prior) !== canonical(r)) {
          conflictKeys.add(key);
          snapshot.issues.push(`Conflicting result excluded: ${r.run_id} / ${b.id}`);
        } else results.set(key, r);
      } catch (error) {
        snapshot.issues.push(`Invalid result ${path.basename(file)}: ${error instanceof z.ZodError ? 'unsupported or malformed format' : String(error)}`);
      }
    }
  }
  if (!validDirs) throw new Error('No readable result directories. Pass --results-dir with an existing directory.');
  const identities = new Map<string, string>();
  const invalidRuns = inconsistentRows;
  for (const r of results.values()) {
    if (publishedRows.has(r.run_id)) continue;
    const identity = canonical([r.model, r.provenance, r.evaluator_version]);
    if (identities.has(r.run_id) && identities.get(r.run_id) !== identity) invalidRuns.add(r.run_id);
    identities.set(r.run_id, identity);
  }
  for (const run of invalidRuns) snapshot.issues.push(`Run has inconsistent model metadata and is excluded: ${run}`);
  for (const [key, r] of results) {
    if (conflictKeys.has(key) || invalidRuns.has(r.run_id)) continue;
    const { predictions: _predictions, environment: _environment, format_version: _version, ...summary } = r;
    const display = presentation.get(key);
    const metadata = display?.metadata;
    const metadataCounts = metadata && (metadata.total_params != null || metadata.active_params != null);
    snapshot.results.push({ ...summary,
      model: metadata ? { ...summary.model, display_name: metadata.display_name, short_name: metadata.short_name,
        url: metadata.url, hf_url: metadata.hf_url,
        total_params: metadataCounts ? metadata.total_params : summary.model.total_params,
        active_params: metadataCounts ? metadata.active_params : summary.model.active_params,
        parameter_count_method: metadataCounts ? metadata.parameter_count_method : summary.model.parameter_count_method } : summary.model,
      original_run_id: display?.original_run_id,
      evaluator_revision: typeof r.environment.evaluator_revision === 'string' ? r.environment.evaluator_revision : undefined,
      dataset_source: datasetSourceSchema.safeParse(r.environment.dataset_source).data,
      scoring: contexts.get(key),
      metrics: { ...summary.metrics, ...computed.get(key) } });
  }
  return snapshot;
}

const shared = globalThis as typeof globalThis & { s1mbSnapshot?: Promise<Snapshot>; s1mbHubResults?: HubResultsCache };
export function getSnapshot(): Promise<Snapshot> {
  const dataDir = process.env.S1MB_DATA_DIR;
  if (!dataDir) throw new Error('Start with npm start so the runtime data directory is configured.');
  const hub = hubResultsOptions();
  if (hub) {
    if (process.env.S1MB_RESULTS_DIRS) throw new Error('Choose either Hub results or explicit local result directories');
    shared.s1mbHubResults ??= new HubResultsCache(hub, directory => loadSnapshot(dataDir, [directory]));
    return shared.s1mbHubResults.get();
  }
  const dirs = process.env.S1MB_RESULTS_DIRS ? JSON.parse(process.env.S1MB_RESULTS_DIRS) as string[] : [path.join(dataDir, 'results')];
  return shared.s1mbSnapshot ??= loadSnapshot(dataDir, dirs);
}
