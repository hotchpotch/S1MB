import assert from "node:assert/strict";
import { test } from "node:test";
import { compareRuns, modelName, hasCompleteCoverage, compareResultMetric, comparisonSpread, comparisonDelta } from "./comparison";
import { METRICS, TASKS, type Snapshot, type ResultSummary } from "./types";
const benchmarks = TASKS.map((task) => ({
  id: `${task}-test`,
  task,
  dataset: "test",
  split: "test",
  case_count: 1,
  decision_count: 1,
  primary_metric: METRICS[task].name,
}));
const category = {
  id: "test",
  name: "Test",
  description: "",
  benchmarks: benchmarks.map((b) => b.id),
};
function result(
  run: string,
  b: (typeof benchmarks)[number],
  value: number,
  extra: Partial<ResultSummary> = {},
): ResultSummary {
  return {
    run_id: run,
    benchmark: b,
    model: { id: "same-model", adapter: "test", revision: "one", settings: {} },
    created_at: "",
    evaluator_version: "1",
    provenance: "measured",
    status: "complete",
    counts: { cases: 1, succeeded: 1, expected: 1, failed: 0 },
    metrics: { [b.primary_metric]: value },
    elapsed_seconds: 1,
    ...extra,
  };
}
function snapshot(results: ResultSummary[]): Snapshot {
  return {
    benchmarks,
    categories: [category],
    results,
    issues: [],
    sources: [],
  };
}

test("Only checked run IDs appear, in selection order; all tasks and missing cells remain", () => {
  const data = snapshot([
    result("choice-only", benchmarks[0], 0.7),
    result("noul-only", benchmarks[1], 0.1),
    result("unchecked", benchmarks[2], 0.2),
  ]);
  const c = compareRuns(data, category, [
    "noul-only",
    "choice-only",
    "noul-only",
    "unknown",
  ]);
  assert.deepEqual(
    c.runs.map((r) => r.id),
    ["noul-only", "choice-only"],
  );
  assert.deepEqual(
    c.groups.map((g) => g.task),
    TASKS,
  );
  assert.equal(c.groups[0].rows[0].cells[0].value, null);
  assert.equal(c.groups[0].rows[0].cells[1].value, 0.7);
  assert.equal(c.groups[2].rows[0].cells.length, 2);
  assert.ok(c.groups[2].rows[0].cells.every((c) => c.value === null));
  assert.equal(compareRuns(data, category, []).runs.length, 0);
});

test("Best values follow each task direction; partial and demo results cannot win", () => {
  const data = snapshot(
    benchmarks.flatMap((b) => [
      result("a", b, 0.8),
      result("b", b, 0.2),
      result("partial", b, b.task === "choice" ? 1 : 0, { status: "partial" }),
      result("demo", b, b.task === "choice" ? 1 : 0, { provenance: "demo" }),
    ]),
  );
  const c = compareRuns(data, category, ["a", "b", "partial", "demo"]);
  assert.deepEqual(
    c.groups.map((g) =>
      g.rows[0].cells.filter((c) => c.best).map((c) => c.runId),
    ),
    [["a"], ["b"], ["b"]],
  );
  assert.equal(c.groups[0].rows[0].cells[2].value, 1);
});

test("Zero scores and ties remain valid; benchmarks outside the category are excluded", () => {
  const data = snapshot([
    result("a", benchmarks[1], 0),
    result("b", benchmarks[1], 0),
  ]);
  const c = compareRuns(data, { ...category, benchmarks: [benchmarks[1].id] }, [
    "a",
    "b",
  ]);
  assert.equal(c.groups[0].rows.length, 0);
  assert.ok(c.groups[1].rows[0].cells.every((c) => c.value === 0 && c.best));
});

test("Model display names support checkpoint labels as well as paths", () => {
  const model = {
    id: "50pct-lr2e4-head2e4",
    adapter: "system-ichi",
    revision: "one",
    settings: {},
  };
  assert.equal(modelName(model), "System Ichi · 50pct-lr2e4-head2e4");
  assert.equal(
    modelName({ ...model, id: "/models/50pct-lr2e4-head2e4/checkpoint" }),
    modelName(model),
  );
});

test('Adjusted comparison uses higher-is-better for every task and preserves undefined values', () => {
  const b = benchmarks[1];
  const data = snapshot([
    result('a', b, 0.1, { metrics: { binary_brier: 0.1, baseline_adjusted_score: 0.2 } }),
    result('b', b, 0.2, { metrics: { binary_brier: 0.2, baseline_adjusted_score: 0.8 } }),
    result('undefined', b, 0, { metrics: { binary_brier: 0, baseline_adjusted_score: null } }),
  ]);
  const c = compareRuns(data, category, ['a', 'b', 'undefined'], true).groups[1].rows[0].cells;
  assert.deepEqual(c.map(c => c.best), [false, true, false]);
  assert.equal(c[2].value, null);
});


test('Model labels preserve checkpoint identity without experiment naming rules', () => {
  const model = { id: 'example/bekko-small', adapter: 'bekko', revision: 'fixture', settings: {} };
  assert.equal(modelName(model), model.id);
});


test('Benchmark metric ranking respects direction, zero, missing values and partial coverage', () => {
  const rows = [
    result('high-f1', benchmarks[1], 0.4, { metrics: { f1: 0.9, binary_brier: 0.4 } }),
    result('zero-error', benchmarks[1], 0, { metrics: { f1: 0.5, binary_brier: 0 } }),
    result('undefined', benchmarks[1], 0.2, { metrics: { f1: null, binary_brier: null } }),
    result('partial', benchmarks[1], 0, { status: 'partial', metrics: { f1: 1, binary_brier: 0 } }),
  ];
  assert.deepEqual([...rows].sort((a,b) => compareResultMetric(a,b,'f1')).map(r => r.run_id), ['high-f1','zero-error','undefined','partial']);
  assert.deepEqual([...rows].sort((a,b) => compareResultMetric(a,b,'binary_brier')).map(r => r.run_id), ['zero-error','high-f1','undefined','partial']);
});

test("Comparison spread excludes partial, demo and missing results while preserving ties", () => {
  const data = snapshot([
    result("a", benchmarks[0], 0.2), result("b", benchmarks[0], 0.7),
    result("partial", benchmarks[0], 1, { status: "partial" }),
    result("demo", benchmarks[0], 0, { provenance: "demo" }),
  ]);
  const row = compareRuns(data, category, ["a", "b", "partial", "demo"]).groups[0].rows[0];
  assert.ok(Math.abs(comparisonSpread(row)! - 0.5) < 1e-9);
  assert.equal(comparisonSpread({ ...row, cells: row.cells.filter(c => c.runId !== "b") }), null);
  assert.equal(comparisonSpread({ ...row, cells: row.cells.slice(0,2).map(c => ({ ...c, value: 0 })) }), 0);
});

test("Comparison deltas preserve direction and reject incomplete references", () => {
  const rows = compareRuns(snapshot([result("a", benchmarks[1], 0.1), result("b", benchmarks[1], 0.3)]), category, ["a", "b"]).groups[1].rows[0];
  assert.ok(Math.abs(comparisonDelta(rows.cells[0], rows.cells[1], false)! - 0.2) < 1e-9);
  assert.ok(Math.abs(comparisonDelta(rows.cells[0], rows.cells[1], true)! + 0.2) < 1e-9);
  assert.equal(comparisonDelta(rows.cells[0], undefined, true), null);
  assert.equal(comparisonDelta(rows.cells[0], { ...rows.cells[1], result: { ...rows.cells[1].result!, status: "partial" } }, true), null);
});


test('leaderboard coverage requires all active benchmarks, including when no model is complete', () => {
  const complete = benchmarks.map(b => result('complete', b, 0.5));
  assert.equal(hasCompleteCoverage(category, complete), true);
  assert.equal(hasCompleteCoverage(category, complete.slice(1)), false);
  assert.equal(hasCompleteCoverage(category, []), false);
  assert.equal(hasCompleteCoverage(category, complete.map((r, i) => i ? r : { ...r, status: 'partial' })), false);
  assert.equal(hasCompleteCoverage(category, complete.map(r => ({ ...r, provenance: 'demo' }))), false);
  assert.equal(hasCompleteCoverage({ ...category, benchmarks: [] }, complete), false);
});

test('metric ranks preserve ties and exclude missing or nonfinite scores', async () => {
  const { metricRanks } = await import('./comparison');
  const entries = [{ id: 'a', value: 0.8 }, { id: 'b', value: 0.8 }, { id: 'c', value: 0 }, { id: 'missing', value: null }, { id: 'invalid', value: NaN }];
  assert.deepEqual([...metricRanks(entries)], [['a', 1], ['b', 1], ['c', 3]]);
  assert.deepEqual([...metricRanks(entries, true)], [['c', 1], ['a', 2], ['b', 2]]);
});
