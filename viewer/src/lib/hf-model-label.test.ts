import assert from 'node:assert/strict';
import test from 'node:test';
import { hfModelLabel } from './hf-model-label';

test('Hub labels retain checkpoint directories while excluding revision and URL suffixes', () => {
  assert.equal(hfModelLabel('https://huggingface.co/example/model'), 'example/model');
  assert.equal(hfModelLabel('https://huggingface.co/example/model/tree/release-v2'), 'example/model');
  assert.equal(hfModelLabel('https://huggingface.co/example/model/tree/main/checkpoints/small?foo=bar#files'), 'example/model/checkpoints/small');
  assert.equal(hfModelLabel('https://huggingface.co/example/model/tree/main/checkpoint%20one'), 'example/model/checkpoint one');
});

test('Only explicit valid Hub model URLs supply a label', () => {
  for (const value of [undefined, null, '', 'invalid', 'https://example.com/org/model',
    'https://huggingface.co/spaces/org/space', 'https://huggingface.co/datasets/org/data',
    'https://huggingface.co/example', 'https://huggingface.co/example/%ZZ']) {
    assert.equal(hfModelLabel(value), null);
  }
});
