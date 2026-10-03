import test from 'node:test';
import assert from 'node:assert/strict';
import { matchesParameterRange, parameterValue } from './parameter-filters';

test('unrestricted ranges include unknown and counts beyond the scale', () => {
  for (const count of [undefined, null, 17e6, 100e9]) assert.equal(matchesParameterRange(count, [0, 100]), true);
});
test('bounded ranges exclude missing counts and include exact boundaries', () => {
  for (const count of [undefined, null, NaN, -1, 400e6, 8e9]) assert.equal(matchesParameterRange(count, [20, 40]), false);
  for (const count of [1e9, 3e9, 5e9]) assert.equal(matchesParameterRange(count, [20, 40]), true);
});
test('endpoints remain open and interior anchors match the displayed scale', () => {
  assert.deepEqual([0, 20, 40, 60, 80, 100].map(parameterValue), [50e6, 1e9, 5e9, 10e9, 20e9, 35e9]);
  assert.equal(matchesParameterRange(17e6, [0, 20]), true);
  assert.equal(matchesParameterRange(100e9, [80, 100]), true);
});
