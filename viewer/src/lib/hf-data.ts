/** Read Hugging Face save_to_disk Arrow shards directly, without a Python service. */
import { readFile, realpath } from 'node:fs/promises';
import path from 'node:path';
import { tableFromIPC, compressionRegistry, CompressionType } from 'apache-arrow';
import { zstdDecompressSync } from 'node:zlib';

compressionRegistry.set(CompressionType.ZSTD, { decode: data => zstdDecompressSync(data) });
import { z } from 'zod';
import type { ScoringCase } from './diagnostics';

type Decision = {
  id: string; kind: string; type: string | null; scoring: string | null;
  criteria: Iterable<{ id: string; value: number | bigint | null }>;
  documents: Iterable<unknown>;
};
type Target = { decision_id: string; kind: string; ids: Iterable<string>; probabilities: Iterable<number> };
type Row = { case_id: string; input_hash?: string; split: string; input: { decisions: Iterable<Decision> }; targets: Iterable<Target> };

/** Extract scoring inputs only; source metadata and instructions never reach the viewer. */
export function scoringCaseFromRow(row: Row): ScoringCase {
  const decisions = Array.from(row.input.decisions);
  const targetList = Array.from(row.targets);
  const byId = new Map(targetList.map(t => [t.decision_id, t]));
  const ids = decisions.map(d => d.id);
  if (new Set(ids).size !== ids.length || byId.size !== targetList.length || ids.length !== byId.size || ids.some(id => !byId.has(id))) throw new Error('Every evaluation decision must have exactly one target');
  const questions: ScoringCase['questions'] = [];
  const targets: ScoringCase['targets'] = {};
  for (const d of decisions) {
    if (d.kind !== 'judgment') throw new Error('Ranking requires ranking metrics; S1MB evaluates judgments only');
    if (Array.from(d.documents).length || d.scoring != null) throw new Error('Judgments cannot contain ranking documents or relative scoring');
    const task = z.enum(['choice', 'noul', 'score']).parse(d.type);
    const criteria = Array.from(d.criteria);
    const candidates = new Map(criteria.map(c => [c.id, c]));
    const order = criteria.map(c => c.id);
    if (candidates.size !== criteria.length || order.length !== criteria.length || new Set(order).size !== order.length || order.some(id => !candidates.has(id))) throw new Error('Invalid criterion IDs or rendering order');
    const target = byId.get(d.id)!;
    if (target.kind !== 'judgment_distribution') throw new Error('Judgment requires a judgment_distribution target');
    const targetIds = Array.from(target.ids), values = Array.from(target.probabilities);
    if (targetIds.length !== values.length || new Set(targetIds).size !== targetIds.length || targetIds.length !== criteria.length || targetIds.some(id => !candidates.has(id))) throw new Error('Invalid target IDs');
    if (values.some(p => !Number.isFinite(p) || p < 0 || p > 1) || Math.abs(values.reduce((a,b) => a+b, 0) - 1) > 1e-5) throw new Error('Invalid target probabilities');
    const options = order.map(id => ({ id, value: candidates.get(id)!.value == null ? null : Number(candidates.get(id)!.value) }));
    if (options.length < 2) throw new Error('Judgments require at least two criteria');
    if (task === 'noul' && (options.length !== 2 || !candidates.has('false') || !candidates.has('true'))) throw new Error('Noul requires false/true criteria');
    if (task === 'score' && (options.some(o => o.value == null || !Number.isFinite(o.value)) || new Set(options.map(o => o.value)).size < 2)) throw new Error('Score requires a nonconstant numeric scale');
    questions.push({ id: d.id, task, options });
    Object.defineProperty(targets, d.id, { value: { probabilities: Object.fromEntries(targetIds.map((id, i) => [id, values[i]])) }, enumerable: true });
  }
  return { case_id: row.case_id, ...(row.input_hash ? { input_hash: row.input_hash } : {}), questions, targets };
}

export async function loadHfCases(dataDir: string, dataset: string, split: string): Promise<ScoringCase[]> {
  const root = await realpath(path.join(dataDir, 'datasets'));
  const directory = await realpath(path.resolve(dataDir, dataset));
  if (!directory.startsWith(root + path.sep)) throw new Error('Dataset path escapes datasets directory');
  const dict = z.object({ splits: z.array(z.string()) }).parse(JSON.parse(await readFile(path.join(directory, 'dataset_dict.json'), 'utf8')));
  if (!dict.splits.includes(split) || !/^[A-Za-z0-9_-]+$/.test(split)) throw new Error('Missing or invalid dataset split');
  const splitDir = await realpath(path.join(directory, split));
  if (!splitDir.startsWith(directory + path.sep)) throw new Error('Split path escapes dataset directory');
  const state = z.object({
    _data_files: z.array(z.object({ filename: z.string() })).nonempty(),
    _indices_data_files: z.array(z.unknown()).optional(),
  }).parse(JSON.parse(await readFile(path.join(splitDir, 'state.json'), 'utf8')));
  if (state._indices_data_files?.length) throw new Error('Save the dataset with save_to_disk to flatten indices first');
  const cases: ScoringCase[] = [];
  for (const shard of state._data_files) {
    const file = await realpath(path.resolve(splitDir, shard.filename));
    if (!file.startsWith(splitDir + path.sep)) throw new Error('Arrow shard escapes split directory');
    const table = tableFromIPC(await readFile(file));
    for (const item of table) {
      const row = item as unknown as Row;
      if (row.split !== split) throw new Error('Row split differs from requested split');
      cases.push(scoringCaseFromRow(row));
    }
  }
  return cases;
}
