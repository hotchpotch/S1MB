import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultDashboardState, parseDashboardState, serializeDashboardState, resetDashboardState, syncDashboardUrl } from './dashboard-state';

test('model deep links retain sidebar, tab, sort and ordered comparison selection', () => {
  const state = parseDashboardState(new URLSearchParams('category=japanese-v1&view=compare&task=noul&run=org%2Fmodel&compare=b&compare=a&compare=b&sort=general-score&ascending=1'));
  assert.equal(state.categoryId, 'japanese-v1');
  assert.equal(state.runId, 'org/model');
  assert.equal(state.view, 'compare');
  assert.equal(state.task, 'noul');
  assert.equal(state.leaderboardSort, 'general-score');
  assert.equal(state.leaderboardAscending, true);
  assert.deepEqual(state.selectedIds, ['b', 'a']);
  assert.deepEqual(parseDashboardState(serializeDashboardState(state)), state);
});

test('benchmark ranking, ranges, comparison settings and sidebar filters survive sharing', () => {
  const params = new URLSearchParams('category=english-v1&view=benchmarks&task=score&benchmark=example-score&metric=normalized_expected_score_mae&query=math&modelSearch=small&generalOnly=1&checkedOnly=1&filtersOpen=1&totalRange=20,80&activeRange=0,40&incomplete=1&pickerClosed=1&comparisonQuery=logic&comparisonSort=name&delta=1&reference=model-b&raw=1&detailQuery=example&detailTask=score');
  const state = parseDashboardState(params);
  assert.equal(state.benchmarkMetric, 'normalized_expected_score_mae');
  assert.deepEqual(state.totalRange, [20, 80]);
  assert.equal(state.comparisonAdjusted, false);
  assert.equal(state.detailTask, 'score');
  assert.deepEqual(parseDashboardState(serializeDashboardState(state)), state);
  for (const [key, value] of params) assert.equal(serializeDashboardState(state).get(key), value);
});

test('invalid enums, flags, metrics and ranges fall back to usable defaults', () => {
  assert.deepEqual(parseDashboardState(new URLSearchParams('category=private&task=oops&view=script&sort=unknown&metric=binary_brier&ascending=true&totalRange=80,20&activeRange=NaN,100&detailTask=invalid&comparisonSort=invalid')), defaultDashboardState());
  for (const value of ['-1,50', '0,101', '0,20,40', ',100', '0,Infinity']) assert.deepEqual(parseDashboardState(new URLSearchParams({ totalRange: value })).totalRange, [0, 100]);
});

test('home resets all controls while preserving English or Japanese scope', () => {
  for (const categoryId of ['english-v1', 'japanese-v1']) {
    const state = parseDashboardState(new URLSearchParams(`category=${categoryId}&view=compare&run=x&compare=x&sort=score&query=x&filtersOpen=1&raw=1`));
    const reset = resetDashboardState(state);
    assert.deepEqual(reset, { ...defaultDashboardState(), categoryId });
    assert.equal(serializeDashboardState(reset).toString(), `category=${categoryId}`);
  }
});

test('serialization preserves unrelated parameters and removes stale state without changing its input', () => {
  const base = new URLSearchParams('embed=1&compare=old&run=old&sort=score&raw=1');
  const params = serializeDashboardState({ ...defaultDashboardState(), selectedIds: ['b / 2', 'a&1'] }, base);
  assert.equal(params.get('embed'), '1');
  for (const key of ['run', 'sort', 'raw']) assert.equal(params.has(key), false);
  assert.deepEqual(params.getAll('compare'), ['b / 2', 'a&1']);
  assert.equal(base.get('run'), 'old');
});

test('HF synchronization uses the official target origin and preserves comparison anchors', () => {
  const messages: unknown[] = [], urls: unknown[] = [];
  const parent = { postMessage: (...args: unknown[]) => messages.push(args) };
  const browser = { parent, location: { hostname: 'synthetic.hf.space', search: '?old=1', hash: '#compare-noul', pathname: '/' }, history: { replaceState: (...args: unknown[]) => urls.push(args) } };
  syncDashboardUrl(browser, new URLSearchParams('category=japanese-v1&run=model'));
  assert.deepEqual(urls, [[null, '', '/?category=japanese-v1&run=model#compare-noul']]);
  assert.deepEqual(messages, [[{ queryString: '?category=japanese-v1&run=model', hash: '#compare-noul' }, 'https://huggingface.co']]);
  messages.length = 0;
  syncDashboardUrl({ ...browser, location: { ...browser.location, hostname: 'localhost' } }, new URLSearchParams());
  assert.equal(messages.length, 0);
});

test('generalization view retains the underlying task and metric when returning to benchmark tabs', () => {
  const state = { ...defaultDashboardState(), task: 'noul' as const, generalizationView: true, benchmarkMetric: 'binary_brier' };
  assert.deepEqual(parseDashboardState(serializeDashboardState(state)), state);
});

test('comparison model sorting restores its heading and direction separately from benchmark row order', () => {
  const state = parseDashboardState(new URLSearchParams('view=compare&compare=b&compare=a&modelSort=general-noul&modelAscending=1&comparisonSort=name'));
  assert.equal(state.modelOrderSort, 'general-noul');
  assert.equal(state.modelOrderAscending, true);
  assert.equal(state.comparisonSort, 'name');
  assert.deepEqual(parseDashboardState(serializeDashboardState(state)), state);
  assert.equal(parseDashboardState(new URLSearchParams('modelSort=borda')).modelOrderSort, null);
});

test('training disclosure focus survives sharing and clears when model details close', async () => {
  const { selectRunDetails } = await import('./dashboard-state');
  for (const training of ['all', 'example-choice']) {
    const state = parseDashboardState(new URLSearchParams(`run=model-a&training=${training}`));
    assert.equal(state.trainingFocus, training);
    assert.deepEqual(parseDashboardState(serializeDashboardState(state)), state);
    const closed = selectRunDetails(state, '');
    assert.equal(closed.trainingFocus, '');
    assert.equal(serializeDashboardState(closed).has('training'), false);
    assert.equal(selectRunDetails(state, 'model-b').trainingFocus, '');
    assert.equal(selectRunDetails(state, 'model-b', 'all').trainingFocus, 'all');
    assert.equal(resetDashboardState(state).trainingFocus, '');
  }
  for (const params of ['training=all', 'run=x&training=../../x', 'run=x&training=']) {
    assert.equal(parseDashboardState(new URLSearchParams(params)).trainingFocus, '');
  }
});
