import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { benchmarkSources } from './benchmark-sources';

test('Active benchmark definitions link to public sources without loading evaluation data', async () => {
  const root = new URL('../../../evaluator/data/', import.meta.url);
  const category = JSON.parse(await readFile(new URL('categories/english-v1.json', root), 'utf8'));
  for (const id of category.benchmarks) {
    const benchmark = JSON.parse(await readFile(new URL(`benchmarks/${id}.json`, root), 'utf8'));
    const links = benchmarkSources(benchmark.dataset);
    assert.ok(links.length > 0, id);
    assert.equal(new Set(links).size, links.length);
    for (const link of links) {
      const url = new URL(link);
      assert.equal(url.protocol, 'https:');
      assert.ok(['huggingface.co', 'github.com'].includes(url.hostname));
      assert.equal(url.username, '');
      assert.equal(url.password, '');
    }
  }
});

test('AQuA links to its original repository; unknown datasets do not invent a link', () => {
  assert.ok(benchmarkSources('datasets/aqua_rat').includes('https://github.com/google-deepmind/AQuA'));
  assert.deepEqual(benchmarkSources('datasets/unknown-synthetic'), []);
});
