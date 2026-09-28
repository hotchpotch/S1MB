import assert from 'node:assert/strict';
import { test } from 'node:test';
import { existsSync } from 'node:fs';
const datasetTest = !process.env.S1MB_TEST_NO_DATASET && existsSync('data/datasets/hub-source.json') ? test : test.skip;
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { tableFromIPC, tableToIPC } from 'apache-arrow';
import { loadHfCases, scoringCaseFromRow } from './hf-data';
import { prepareScoring } from './diagnostics';

datasetTest('HF shard order comes from state.json; undeclared files are ignored', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 's1mb-arrow-'));
  try {
    const folder = path.join(root, 'datasets/sample/test');
    await mkdir(folder, { recursive: true });
    const table = tableFromIPC(await readFile('data/datasets/arc/test/data-00000-of-00001.arrow'));
    await writeFile(path.join(folder, '../dataset_dict.json'), JSON.stringify({ splits: ['test'] }));
    await writeFile(path.join(folder, 'state.json'), JSON.stringify({ _data_files: [{ filename: 'z.arrow' }, { filename: 'a.arrow' }] }));
    await writeFile(path.join(folder, 'z.arrow'), tableToIPC(table.slice(0, 1)));
    await writeFile(path.join(folder, 'a.arrow'), tableToIPC(table.slice(1, 2)));
    await writeFile(path.join(folder, 'ignored.arrow'), 'invalid');
    const cases = await loadHfCases(root, 'datasets/sample', 'test');
    assert.deepEqual(cases.map(c => c.case_id), [table.get(0)!.case_id, table.get(1)!.case_id]);
    assert.equal(prepareScoring(cases, 'choice').decisions, 2);
    await assert.rejects(loadHfCases(root, 'datasets/sample', 'train'), /Missing/);
    await writeFile(path.join(folder, 'state.json'), JSON.stringify({ _data_files: [{ filename: '../dataset_dict.json' }] }));
    await assert.rejects(loadHfCases(root, 'datasets/sample', 'test'), /escapes/);
  } finally { await rm(root, { recursive: true }); }
});

datasetTest('Nano Score keeps shuffled IDs and numeric scale; repaired CrowS baseline is not perfect', async () => {
  const cases = await loadHfCases('data', 'datasets/synthetic_relevance__nanobeir__NanoMSMARCO', 'test');
  assert.equal(cases.length, 400);
  assert.deepEqual(cases[0].questions[0].options.map(o => o.value).sort(), [0, 1, 2, 3, 4]);
  assert.equal(prepareScoring(cases, 'score').decisions, 400);
  const crows = prepareScoring(await loadHfCases('data', 'datasets/crows_pairs', 'test'), 'choice');
  assert.equal(crows.calculate([]).fixed_answer_accuracy_baseline, 0.52);
  assert.equal(crows.context.eligible, true);
});

test('Structured judgments align teachers by ID and reject ranking and malformed targets', async () => {
  const row = JSON.parse(await readFile('../evaluator/tests/fixtures/system_one.json', 'utf8'));
  const value = scoringCaseFromRow(row);
  assert.deepEqual(value.questions.map(q => q.task), ['choice', 'noul', 'score']);
  assert.deepEqual(value.questions[2].options, [{ id: 'high-id', value: 0 }, { id: 'low-id', value: 4 }]);
  assert.deepEqual(value.targets.quality.probabilities, { 'low-id': 0.75, 'high-id': 0.25 });
  assert.deepEqual(Object.keys(value).sort(), ['case_id', 'input_hash', 'questions', 'targets']);
  for (const change of [
    (r: typeof row) => { r.input.decisions[0].kind = 'ranking'; },
    (r: typeof row) => { r.targets.pop(); },
    (r: typeof row) => { r.targets.push(r.targets[0]); },
    (r: typeof row) => { r.targets[0].kind = 'ranking_distribution'; },
    (r: typeof row) => { r.targets[0].ids = ['low-id', 'low-id']; },
    (r: typeof row) => { r.targets[0].probabilities = [-1, 2]; },
  ]) {
    const bad = structuredClone(row);
    change(bad);
    assert.throws(() => scoringCaseFromRow(bad));
  }
});
