import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { DataType, Field, Float64, List, RecordBatch, Schema, Struct, Table, Utf8, vectorFromArray, tableToIPC } from 'apache-arrow';
import { loadSnapshot } from './results';
import { scoringCaseFromRow } from './hf-data';
import { prepareScoring } from './diagnostics';
import { categoryRuns, modelName } from './comparison';
import { diagnosticMean, type Task } from './types';

async function put(file: string, value: unknown) {
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify(value));
}
async function compressed(file: string, value: unknown) {
  await mkdir(path.dirname(file), { recursive: true });
  const bytes = execFileSync('xz', ['--compress', '--stdout'], { input: JSON.stringify(value) });
  await writeFile(file, bytes);
}
async function dataset(root: string, row: Record<string, unknown>) {
  const folder = path.join(root, 'datasets/sample/test');
  await mkdir(folder, { recursive: true });
  await put(path.join(folder, '../dataset_dict.json'), { splits: ['test'] });
  await put(path.join(folder, 'state.json'), { _data_files: [{ filename: 'data.arrow' }] });
  const field = (name: string, type: DataType) => new Field(name, type, true);
  const list = (type: DataType) => new List(field('item', type));
  const text = () => new Utf8();
  const decision = new Struct([field('id', text()), field('kind', text()), field('type', text()), field('scoring', text()),
    field('documents', list(text())), field('criteria', list(new Struct([field('id', text()), field('value', new Float64())])))]);
  const target = new Struct([field('decision_id', text()), field('kind', text()), field('ids', list(text())), field('probabilities', list(new Float64()))]);
  const schema = new Struct([field('case_id', text()), field('input_hash', text()), field('split', text()),
    field('input', new Struct([field('decisions', list(decision))])), field('targets', list(target))]);
  const vector = vectorFromArray([row], schema);
  const table = new Table(new RecordBatch(new Schema(schema.children), vector.data[0]));
  await writeFile(path.join(folder, 'data.arrow'), tableToIPC(table));
}

test('Published model folders combine runs and revisions, support replacement, and retain provenance', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-published-'));
  try {
    const data = path.join(root, 'data'), results = path.join(root, 'results');
    const folder = path.join(results, 'test__model_v1');
    const row = JSON.parse(await readFile('../evaluator/tests/fixtures/system_one.json', 'utf8'));
    const source = { repo_id: 'test/dataset', revision: 'a'.repeat(40) };
    const oldSource = { ...source, revision: 'b'.repeat(40) };
    const oldRow = structuredClone(row);
    oldRow.input_hash = 'older-input';
    // Different targets must be scored against the recorded release.
    oldRow.targets.find((target: { decision_id: string }) => target.decision_id === 'label').probabilities = [0.25, 0.75];
    await dataset(data, row);
    await put(path.join(data, 'datasets/hub-source.json'), source);
    await dataset(path.join(data, 'result-datasets', createHash('sha256').update(`${oldSource.repo_id}@${oldSource.revision}`).digest('hex')), oldRow);
    await put(path.join(folder, 'metadata.json'), { model_id: 'test__model_v1', display_name: 'Example model', short_name: 'Example', hf_url: 'https://huggingface.co/test/model' });
    const benchmarks = [];
    for (const [index, task] of (['choice', 'noul'] as Task[]).entries()) {
      const currentRow = index === 0 ? oldRow : row;
      const cases = [scoringCaseFromRow(currentRow)];
      const prepared = prepareScoring(cases, task);
      const benchmark = { id: `sample-${task}`, task, dataset: 'datasets/sample', split: 'test', case_count: 1, decision_count: 1,
        primary_metric: task === 'choice' ? 'target_mass_at_prediction' : 'binary_brier' };
      benchmarks.push(benchmark);
      await put(path.join(data, 'benchmarks', `${benchmark.id}.json`), benchmark);
      const predictions = cases.flatMap(c => c.questions.filter(q => q.task === task).map(q => ({ case_id: c.case_id, question_id: q.id,
        probabilities: Object.fromEntries(q.options.map((o, i) => [o.id, i === 0 ? 1 : 0])), error: null })));
      await compressed(path.join(folder, `${benchmark.id}.json.xz`), { format_version: 1, run_id: `run-${index}`, benchmark,
        model: { id: 'test/model', adapter: 'fixture', revision: `model-${index}`, settings: {} },
        created_at: '2026-09-29T00:00:00Z', evaluator_version: `0.1.${index}`, provenance: 'measured', status: 'complete',
        counts: { cases: 1, expected: 1, succeeded: 1, failed: 0 }, metrics: prepared.calculate(predictions), elapsed_seconds: 0,
        environment: { input_hashes: { [currentRow.case_id]: currentRow.input_hash }, dataset_source: index === 0 ? oldSource : source }, predictions });
    }
    await put(path.join(data, 'categories/test.json'), { id: 'test', name: 'Test', description: 'Synthetic', benchmarks: benchmarks.map(b => b.id) });
    let snapshot = await loadSnapshot(data, [results]);
    assert.deepEqual(snapshot.issues, []);
    const runs = categoryRuns(snapshot, snapshot.categories[0]);
    assert.equal(runs.length, 1);
    assert.equal(runs[0].id, 'test__model_v1');
    assert.equal(modelName(runs[0].model), 'Example model');
    assert.equal(runs[0].results.length, 2);
    assert.deepEqual(runs[0].results.map(r => r.original_run_id), ['run-0', 'run-1']);
    assert.equal(runs[0].results[0].dataset_source?.revision, oldSource.revision);
    assert.equal('predictions' in runs[0].results[0], false);
    assert.notEqual(diagnosticMean(snapshot, snapshot.categories[0], 'choice', runs[0].results, 'baseline_adjusted_score'), null);
    // Replacing one file changes one cell without producing another leaderboard row.
    const file = path.join(folder, 'sample-choice.json.xz');
    const changed = JSON.parse(execFileSync('xz', ['-dc', file], { encoding: 'utf8' }));
    changed.run_id = 'replacement';
    await compressed(file, changed);
    snapshot = await loadSnapshot(data, [results]);
    assert.equal(snapshot.results.length, 2);
    assert.equal(snapshot.results[0].original_run_id, 'replacement');
    changed.environment.input_hashes = {};
    await compressed(file, changed);
    snapshot = await loadSnapshot(data, [results]);
    assert.equal(snapshot.results.length, 1);
    assert.match(snapshot.issues.join(), /input hashes differ/);
    assert.equal(diagnosticMean(snapshot, snapshot.categories[0], 'choice', snapshot.results, 'baseline_adjusted_score'), null);
    await put(path.join(folder, 'metadata.json'), { model_id: 'test__other', display_name: 'Bad', short_name: 'Bad' });
    snapshot = await loadSnapshot(data, [results]);
    assert.equal(snapshot.results.length, 0);
    assert.match(snapshot.issues.join(), /metadata ID mismatch/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
