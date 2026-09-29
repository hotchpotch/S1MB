"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowLeftRight, ArrowUpRight, Check, ChevronLeft, ChevronRight, Search, X } from "lucide-react";
import { Dialog } from "radix-ui";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Checkbox } from "./ui/checkbox";
import { Input } from "./ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs";
import { Separator } from "./ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import {
  leaderboard,
  diagnosticMean,
  overallIndex,
  instructionLabel,
  METRICS,
  TASKS,
  DISPLAY_TASKS,
  generalizationCategory,
  type Snapshot,
  type Task,
  type ModelInfo,
  type ResultSummary,
} from "../lib/types";
import {
  benchmarkName,
  categoryRuns,
  compareRuns,
  compareResultMetric,
  comparisonSpread,
  comparisonDelta,
  modelName,
  type Run,
} from "../lib/comparison";
import { cn } from "../lib/utils";

type View = "leaderboard" | "benchmarks" | "compare";
const adjustedScore = (value: number | null | undefined) => value == null ? '—' : (value * 100).toFixed(2);
const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(2)}%`;
const BENCHMARK_METRICS = {
  choice: [{ name: 'target_mass_at_prediction', label: 'Target mass ↑' }],
  noul: [
    { name: 'f1', label: 'F1 ↑' },
    { name: 'precision', label: 'Precision ↑' },
    { name: 'positive_recall', label: 'Recall ↑' },
    { name: 'balanced_accuracy', label: 'Balanced acc. ↑' },
    { name: 'binary_brier', label: 'Brier ↓' },
  ],
  score: [
    { name: 'normalized_expected_score_mae', label: 'MAE ↓' },
    { name: 'normalized_score_rmse', label: 'RMSE ↓' },
  ],
};
const title = (s: string) => s[0].toUpperCase() + s.slice(1);
const repeatedName = (runs: Run[], model: ModelInfo) => runs.filter(r => modelName(r.model) === modelName(model)).length > 1;
const direction = (task: Task) =>
  METRICS[task].direction === "up" ? "↑ Higher is better" : "↓ Lower is better";

export function CoverageBadge({
  complete,
  total,
  demo = false,
}: {
  complete: number;
  total: number;
  demo?: boolean;
}) {
  return (
    <Badge
      variant={demo || complete !== total ? "secondary" : "outline"}
      className={cn(
        "whitespace-nowrap",
        !demo && complete === total && "text-primary border-primary/30",
      )}
    >
      {demo ? "Demo · " : ""}
      {complete}/{total} complete
    </Badge>
  );
}
function ModelLabel({
  model,
  runId,
  onClick,
  detailed = false,
  distinguish = false,
}: {
  model: ModelInfo;
  runId: string;
  onClick?: () => void;
  detailed?: boolean;
  distinguish?: boolean;
}) {
  return (
    <div className="min-w-0 w-full space-y-1 text-left whitespace-normal [overflow-wrap:anywhere]">
      {onClick ? (
        <button
          onClick={onClick}
          className="font-semibold text-sm text-foreground underline-offset-4 hover:underline text-left break-words"
        >
          {modelName(model)}
        </button>
      ) : (
        <p className="font-semibold text-sm break-words">{modelName(model)}</p>
      )}
      {(detailed || (!onClick && instructionLabel(model) !== 'Default instructions')) && <p className="text-xs font-normal text-muted-foreground">{instructionLabel(model)}</p>}
      {(detailed || !onClick) && <details className="text-[11px] font-normal text-muted-foreground" open={detailed || distinguish || undefined}>
        <summary className="cursor-pointer">Run ID</summary>
        <p className="font-mono break-all mt-1">{runId}</p>
      </details>}
    </div>
  );
}
function DiagnosticDetails({ result }: { result: ResultSummary }) {
  const metrics = result.metrics;
  const labels: Record<string, string> = {
    accuracy: 'Accuracy', balanced_accuracy: 'Balanced accuracy', positive_recall: 'True recall',
    specificity: 'False recall', precision: 'Precision', f1: 'F1', positive_prevalence: 'Target true rate',
    majority_accuracy_baseline: 'Majority accuracy baseline', constant_brier_baseline: 'Constant Brier baseline',
    brier_skill: 'Brier skill', uniform_accuracy_baseline: 'Uniform baseline',
    fixed_answer_accuracy_baseline: 'Fixed-answer baseline', normalized_score_rmse: 'Normalized RMSE',
    constant_score_baseline: 'Constant normalized score', constant_mae_baseline: 'Constant MAE baseline',
    baseline_adjusted_skill: 'Adjusted skill (unclipped)',
  };
  return <details className="text-xs mt-2 text-muted-foreground whitespace-normal min-w-40">
    <summary className="cursor-pointer">More metrics</summary>
    <dl className="space-y-1 mt-2">{Object.entries(labels).filter(([name]) => name in metrics).map(([name, label]) =>
      <div key={name} className="flex justify-between gap-3"><dt>{label}</dt><dd className="font-mono">{metrics[name]?.toFixed(4) ?? 'N/A'}</dd></div>)}
    </dl>
    {result.benchmark.task === 'noul' && 'true_positive' in metrics && <p className="mt-2 font-mono">TP {metrics.true_positive?.toFixed(2)} / FN {metrics.false_negative?.toFixed(2)} / FP {metrics.false_positive?.toFixed(2)} / TN {metrics.true_negative?.toFixed(2)}</p>}
  </details>;
}
function MetricGuide({ task, result }: { task: Task; result?: ResultSummary }) {
  const metrics = result?.metrics;
  return <details className="text-sm text-muted-foreground">
    <summary className="cursor-pointer underline underline-offset-4">How to read these metrics</summary>
    <div className="mt-3 max-w-3xl space-y-2 leading-relaxed">
      <p>Adjusted scores compare performance with a constant baseline. They are not accuracy. Higher is better; 0% includes results at or below baseline.</p>
      {task === 'noul' ? <>
        <p>F1 balances precision and recall for true predictions. Precision asks how often a true prediction is correct; recall asks how many true targets were found. Balanced accuracy gives true and false targets equal weight. Brier measures probability error; lower is better.</p>
        {metrics?.positive_prevalence != null && <p>True target share: <strong>{percent(metrics.positive_prevalence)}</strong>. Always choosing the majority label yields {percent(metrics.majority_accuracy_baseline)} accuracy. These values use the complete result shown first.</p>}
      </> : task === 'choice' ? <>
        <p>Target mass measures the reference probability assigned to the chosen answer. Adjusted scores account for the best fixed-answer baseline.</p>
        {metrics?.fixed_answer_accuracy_baseline != null && <p>Fixed-answer baseline: {percent(metrics.fixed_answer_accuracy_baseline)}.</p>}
      </> : <>
        <p>MAE is mean absolute error on the normalized score scale. RMSE weights large errors more strongly. Both are lower-is-better; neither is an accuracy percentage.</p>
        {metrics?.constant_mae_baseline != null && <p>Constant-prediction baseline MAE: {metrics.constant_mae_baseline.toFixed(4)}.</p>}
      </>}
      <p>— means undefined or unavailable, not zero. Partial results do not participate in best-score highlighting.</p>
    </div>
  </details>;
}
function metricValue(name: string, value: number | null | undefined) {
  if (value == null) return '—';
  return ['baseline_adjusted_score', 'target_mass_at_prediction', 'f1', 'precision', 'positive_recall', 'balanced_accuracy'].includes(name) ? percent(value) : value.toFixed(4);
}
function ProfileScore({ value }: { value: number | null }) {
  return <div className="min-w-0 text-right">
    <span className="font-mono tabular-nums font-semibold">{percent(value)}</span>
    {value !== null && <div className="mt-1 h-1 rounded-full overflow-hidden bg-muted" aria-hidden="true"><div className="h-full bg-primary" style={{ width: `${value * 100}%` }} /></div>}
  </div>;
}
function ScoreValue({
  result,
  best = false,
  adjusted = false,
  compact = false,
}: {
  result?: ResultSummary;
  best?: boolean;
  adjusted?: boolean;
  compact?: boolean;
}) {
  const value = result?.metrics[adjusted ? 'baseline_adjusted_score' : result.benchmark.primary_metric];
  return (
    <div className="space-y-1">
      <div
        className={cn(
          "font-mono tabular-nums text-base",
          best && "font-bold text-primary",
        )}
      >
        {value == null ? "—" : adjusted ? percent(value) : value.toFixed(4)}
        {best && (
          <Check
            className="inline size-3.5 ml-1"
            aria-label="Best among selected complete measured results"
          />
        )}
      </div>
      {(!compact || !result || result.status === "partial") && <p className="text-[11px] text-muted-foreground">
        {!result
          ? "Missing"
          : `${result.counts.succeeded.toLocaleString()}/${result.counts.expected.toLocaleString()} decisions`}
      </p>}
      {result?.status === "partial" && (
        <Badge variant="secondary">
          Partial · {result.counts.failed} failed
        </Badge>
      )}
      {result?.provenance === "demo" && <Badge variant="secondary">Demo</Badge>}
      {result && !compact && <DiagnosticDetails result={result} />}
    </div>
  );
}
function CheckRun({
  id,
  checked,
  onToggle,
}: {
  id: string;
  checked: boolean;
  onToggle: (id: string) => void;
}) {
  return (
    <Checkbox
      aria-label={`Compare ${id}`}
      checked={checked}
      onCheckedChange={() => onToggle(id)}
    />
  );
}

export function ComparisonTables({
  snapshot,
  categoryId,
  selectedIds,
  onBenchmarkSelect,
  onRemove,
}: {
  snapshot: Snapshot;
  categoryId: string;
  selectedIds: string[];
  onBenchmarkSelect?: (id: string, task: Task) => void;
  onRemove?: (id: string) => void;
}) {
  const [filter, setFilter] = useState('');
  const [sort, setSort] = useState('gap');
  const [showDelta, setShowDelta] = useState(false);
  const [referenceId, setReference] = useState('');
  const [adjusted, setAdjusted] = useState(true);
  const category = snapshot.categories.find((c) => c.id === categoryId);
  if (!category) return null;
  const comparison = compareRuns(snapshot, category, selectedIds, adjusted);
  const availableRuns = categoryRuns(snapshot, category);
  const reference = comparison.runs.find(r => r.id === referenceId) ?? comparison.runs[0];
  const groups = comparison.groups.map(group => ({ ...group, rows: group.rows.filter(row => `${row.benchmark.id} ${benchmarkName(row.benchmark)}`.toLowerCase().includes(filter.trim().toLowerCase())).sort((a, b) => sort === 'gap' ? (comparisonSpread(b) ?? -1) - (comparisonSpread(a) ?? -1) || a.benchmark.id.localeCompare(b.benchmark.id) : a.benchmark.id.localeCompare(b.benchmark.id)) }));
  const shown = groups.reduce((sum, group) => sum + group.rows.length, 0);
  if (!comparison.runs.length)
    return (
      <Empty
        title="Choose runs to compare"
        description="Check runs in the list above, or select them from the leaderboard. Only checked runs appear here."
      />
    );
  return (
    <div className="space-y-5" aria-label="Selected run comparison">
      <div className="sm:hidden space-y-3" aria-label="Comparison summary cards">
        {comparison.runs.map(run => <article key={run.id} className="rounded-lg border bg-card p-3 space-y-3">
          <div className="flex items-start gap-2"><ModelLabel model={run.model} runId={run.id} distinguish={repeatedName(availableRuns, run.model)} />{onRemove && <Button variant="ghost" size="icon" aria-label={`Remove ${run.id}`} onClick={() => onRemove(run.id)}><X className="size-4" /></Button>}</div>
          <dl className="grid grid-cols-4 gap-2">
            {[{ label: 'Overall', value: overallIndex(snapshot, category, run.results) }, ...TASKS.map(t => ({ label: title(t), value: diagnosticMean(snapshot, category, t, run.results, 'baseline_adjusted_score') }))].map(item => <div key={item.label}><dt className="text-xs text-muted-foreground mb-1">{item.label}</dt><dd><ProfileScore value={item.value} /></dd></div>)}
          </dl>
        </article>)}
        <p className="text-xs text-muted-foreground">Adjusted scores · Fixed 0–100% scale</p>
      </div>
      <div className="hidden sm:block rounded-lg border bg-card overflow-hidden">
        <Table aria-label="Comparison summary" className="min-w-[750px] table-fixed">
          <TableHeader><TableRow><TableHead className="w-[36%]">Selected model</TableHead><TableHead className="text-right">Overall ↑</TableHead>{TASKS.map(t => <TableHead key={t} className="text-right">{title(t)} ↑</TableHead>)}</TableRow></TableHeader>
          <TableBody>{comparison.runs.map(run => <TableRow key={run.id}>
            <TableCell className="py-3 pr-5"><div className="flex items-start gap-2"><ModelLabel model={run.model} runId={run.id} distinguish={repeatedName(availableRuns, run.model)} />{onRemove && <Button variant="ghost" size="icon" aria-label={`Remove ${run.id}`} onClick={() => onRemove(run.id)}><X className="size-3" /></Button>}</div></TableCell>
            <TableCell><ProfileScore value={overallIndex(snapshot, category, run.results)} /></TableCell>
            {TASKS.map(t => <TableCell key={t}><ProfileScore value={diagnosticMean(snapshot, category, t, run.results, 'baseline_adjusted_score')} /></TableCell>)}
          </TableRow>)}</TableBody>
          <caption className="caption-bottom p-3 text-left text-xs text-muted-foreground">Adjusted scores · Fixed 0–100% scale · — means incomplete or unavailable</caption>
        </Table>
      </div>
      <div className="space-y-3">
        {comparison.runs.length > 1 && <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm"><Checkbox checked={showDelta} onCheckedChange={v => setShowDelta(v === true)} />Show difference from reference</label>
          {showDelta && <Select value={reference.id} onValueChange={setReference}><SelectTrigger aria-label="Reference model" className="w-full sm:w-80"><SelectValue /></SelectTrigger><SelectContent>{comparison.runs.map(run => <SelectItem key={run.id} value={run.id}>{modelName(run.model)}</SelectItem>)}</SelectContent></Select>}
          {showDelta && <span className="text-xs text-muted-foreground">+ means better · {adjusted ? 'percentage points (pp)' : 'original metric units'}</span>}
        </div>}
      <label className="flex items-center gap-2 text-sm"><Checkbox checked={adjusted} onCheckedChange={v => setAdjusted(v === true)} />Compare baseline-adjusted scores (higher is better)</label>
        <div className="flex flex-wrap gap-3 items-center">
          <Input aria-label="Filter comparison benchmarks" placeholder="Find a benchmark across all tasks…" value={filter} onChange={event => setFilter(event.target.value)} className="sm:max-w-sm" />
          <Select value={sort} onValueChange={setSort}><SelectTrigger aria-label="Comparison order" className="w-48"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="gap">Largest differences</SelectItem><SelectItem value="name">Benchmark name</SelectItem></SelectContent></Select>
          {filter && <Button variant="ghost" size="sm" onClick={() => setFilter('')}>Clear filter</Button>}
          <nav aria-label="Comparison tasks" className="flex gap-4 text-sm">{groups.filter(g => g.rows.length).map(g => <a key={g.task} href={`#compare-${g.task}`} className="underline underline-offset-4">{title(g.task)} ({g.rows.length})</a>)}</nav>
        </div>
        <p className="text-xs text-muted-foreground" role="status">{shown} / {category.benchmarks.length} benchmarks shown. Gap is the range among complete measured results; it is not statistical significance. Summary uses the full category.</p>
      </div>
      {!shown && <Empty title="No matching benchmarks" description="Clear the filter to return to all tasks." />}
      {groups
        .filter((g) => g.rows.length)
        .map((group) => (
          <section key={group.task} aria-labelledby={`compare-${group.task}`}>
            <div className="flex flex-wrap items-baseline gap-3 mb-3">
              <h3
                id={`compare-${group.task}`}
                className="text-lg font-semibold"
              >
                {title(group.task)}
              </h3>
              <Badge variant="secondary">{group.rows.length} benchmarks</Badge>
              <span className="text-xs text-muted-foreground">
                {adjusted ? 'Baseline-adjusted score · ↑ Higher is better' : `${METRICS[group.task].label} · ${direction(group.task)}`}
              </span>
            </div>
            <div className="rounded-lg border bg-card overflow-hidden">
              <Table aria-label={`${title(group.task)} benchmark comparison`} containerClassName="max-h-[65vh] focus-visible:outline-2 focus-visible:outline-ring" containerLabel={`${title(group.task)} comparison scroll area`} className="table-fixed" style={{ minWidth: 192 + comparison.runs.length * 160 }}>
                <TableHeader>
                  <TableRow>
                    <TableHead className="sticky left-0 top-0 z-30 bg-card w-[128px] sm:w-[220px] whitespace-normal">
                      Benchmark
                    </TableHead>
                    <TableHead className="sticky top-0 z-20 bg-card text-right w-[64px] sm:w-[100px]">Gap {adjusted ? '(pp)' : ''}</TableHead>
                    {comparison.runs.map((run) => (
                      <TableHead
                        key={run.id}
                        className="sticky top-0 z-20 bg-card w-[160px] sm:w-[240px] py-4 px-4 align-top whitespace-normal"
                      >
                        <span title={run.id} className="font-semibold text-foreground [overflow-wrap:anywhere]">{modelName(run.model)}</span>
                        {repeatedName(availableRuns, run.model) && <p className="mt-1 font-mono text-[10px] font-normal break-all">{run.id}</p>}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {group.rows.map(({ benchmark, cells }) => (
                    <TableRow key={benchmark.id}>
                      <TableCell className="sticky left-0 z-10 bg-card whitespace-normal [overflow-wrap:anywhere]">
                        {onBenchmarkSelect ? <button className="font-medium capitalize text-left underline underline-offset-4 whitespace-normal [overflow-wrap:anywhere]" onClick={() => onBenchmarkSelect(benchmark.id, benchmark.task)}>{benchmarkName(benchmark)}</button> : <p className="font-medium capitalize">{benchmarkName(benchmark)}</p>}
                        <p className="text-xs text-muted-foreground mt-1">{benchmark.decision_count.toLocaleString()} decisions</p>
                        {snapshot.scoring?.[benchmark.id]?.reason && <p className="text-xs mt-2 text-muted-foreground">Adjusted N/A: {snapshot.scoring[benchmark.id].reason}</p>}
                      </TableCell>
                      <TableCell className="text-right font-mono text-muted-foreground">{comparisonSpread({ benchmark, cells }) == null ? '—' : adjusted ? (comparisonSpread({ benchmark, cells })! * 100).toFixed(1) : comparisonSpread({ benchmark, cells })!.toFixed(4)}</TableCell>
                      {cells.map((cell) => (
                        <TableCell
                          key={cell.runId}
                          className={cn(
                            "align-top py-4 px-4 whitespace-normal",
                            cell.best && "bg-accent/60",
                          )}
                        >
                          <ScoreValue result={cell.result} best={cell.best} adjusted={adjusted} compact />
                          {showDelta && comparison.runs.length > 1 && <p className="mt-1 text-xs font-mono text-muted-foreground">
                            {cell.runId === reference.id ? 'Reference' : (() => {
                              const delta = comparisonDelta(cell, cells.find(c => c.runId === reference.id), adjusted || METRICS[group.task].direction === 'up');
                              return delta == null ? 'Δ —' : `Δ ${delta > 0 ? '+' : ''}${(delta * (adjusted ? 100 : 1)).toFixed(adjusted ? 1 : 4)}${adjusted ? ' pp' : ''}`;
                            })()}
                          </p>}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </section>
        ))}
    </div>
  );
}
function Empty({
  title: heading,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-lg border border-dashed py-14 px-6 text-center">
      <h3 className="font-semibold">{heading}</h3>
      <p className="text-muted-foreground text-sm mt-2">{description}</p>
    </div>
  );
}

export function Dashboard({
  snapshot,
  initialTask,
  initialRun,
  initialView,
  initialCompare = [],
  initialBenchmark,
}: {
  snapshot: Snapshot;
  initialCategory?: string;
  initialTask?: string;
  initialRun?: string;
  initialView?: string;
  initialCompare?: string[];
  initialBenchmark?: string;
}) {
  const categoryId = "english-v1";
  const [task, setTask] = useState<Task>(
    TASKS.includes(initialTask as Task) ? (initialTask as Task) : "choice",
  );
  const [runId, setRun] = useState(initialRun ?? "");
  const [view, setView] = useState<View>(
    ["leaderboard", "benchmarks", "compare"].includes(initialView ?? "")
      ? (initialView as View)
      : "leaderboard",
  );
  const [selectedIds, setSelected] = useState<string[]>([
    ...new Set(initialCompare),
  ]);
  const [includeIncomplete, setIncomplete] = useState(false);
  const [query, setQuery] = useState("");
  const [benchmarkMetric, setBenchmarkMetric] = useState('baseline_adjusted_score');
  const [benchmarkId, setBenchmark] = useState(initialBenchmark ?? "");
  const category =
    snapshot.categories.find((c) => c.id === categoryId) ??
    snapshot.categories[0];
  useEffect(() => {
    if (!category) return;
    const params = new URLSearchParams(window.location.search);
    for (const key of [
      "category",
      "task",
      "view",
      "run",
      "benchmark",
      "compare",
    ])
      params.delete(key);
    params.set("category", category.id);
    params.set("task", task);
    params.set("view", view);
    if (runId) params.set("run", runId);
    if (benchmarkId) params.set("benchmark", benchmarkId);
    for (const id of selectedIds) params.append("compare", id);
    window.history.replaceState(null, "", `?${params}`);
  }, [category, task, view, runId, selectedIds, benchmarkId]);
  if (!category)
    return (
      <main className="max-w-6xl mx-auto p-8">
        <Empty
          title="No categories"
          description="Add a benchmark category to begin."
        />
      </main>
    );
  const benchmarks = snapshot.benchmarks.filter((b) =>
    category.benchmarks.includes(b.id),
  );
  const generalizationColumns = ['diverse', 'contextual'].flatMap(family =>
    DISPLAY_TASKS.flatMap(t => {
      const benchmark = benchmarks.find(b => b.dataset === `datasets/s1mb-generalization-${family}-${t}`);
      return benchmark ? [{ family, task: t, benchmark }] : [];
    }),
  );
  const runs = categoryRuns(snapshot, category);
  const selectedRuns = selectedIds.flatMap(id => { const run = runs.find(r => r.id === id); return run ? [run] : []; });
  const active = runs.find((r) => r.id === runId);
  const rawRows = leaderboard(snapshot, category, task);
  const allRows = rawRows.map(row => ({ ...row, score: overallIndex(snapshot, category, row.results) }))
    .sort((a, b) => Number(a.demo) - Number(b.demo) || Number(a.score === null) - Number(b.score === null) || (b.score ?? 0) - (a.score ?? 0) || a.runId.localeCompare(b.runId));
  const ranked = allRows.filter((r) => r.score !== null && !r.demo);
  const rows = includeIncomplete || !ranked.length ? allRows : ranked;
  const filteredBenchmarks = benchmarks
    .filter(
      (b) =>
        b.task === task && `${b.id} ${benchmarkName(b)}`.toLowerCase().includes(query.trim().toLowerCase()),
    )
    .sort((a, b) => a.id.localeCompare(b.id));
  const benchmark =
    filteredBenchmarks.find((b) => b.id === benchmarkId) ??
    filteredBenchmarks[0];
  const individual = benchmark
    ? compareRuns(
        snapshot,
        category,
        runs.map((r) => r.id),
      )
        .groups.find((g) => g.task === task)!
        .rows.find((r) => r.benchmark.id === benchmark.id)!
    : undefined;
  const individualCells =
    individual?.cells
      .filter((c) => c.result && (includeIncomplete || c.result.provenance === "measured"))
      .sort((a, b) => compareResultMetric(a.result!, b.result!, benchmarkMetric)) ?? [];
  const metricColumns = [{ name: 'baseline_adjusted_score', label: 'Adjusted ↑' }, ...BENCHMARK_METRICS[task]];
  const sortLabel = metricColumns.find(m => m.name === benchmarkMetric)?.label ?? 'Adjusted ↑';
  const metricDirection = ['binary_brier', 'normalized_expected_score_mae', 'normalized_score_rmse'].includes(benchmarkMetric) ? 'ascending' : 'descending';
  const bestMetric = individualCells.find(c => c.result?.status === 'complete' && c.result.provenance === 'measured' && c.result.metrics[benchmarkMetric] != null)?.result?.metrics[benchmarkMetric];
  function toggle(id: string) {
    setSelected((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id],
    );
  }
  function chooseTask(value: string) {
    setTask(value as Task);
    setBenchmarkMetric('baseline_adjusted_score');
    setBenchmark("");
    setQuery("");
  }
  const visibleRuns = runs
    .filter(
      (r) =>
        includeIncomplete ||
        !runs.some(
          (x) => !x.demo && x.results.some((y) => y.status === "complete"),
        ) ||
        (!r.demo && r.results.some((x) => x.status === "complete")),
    )
    .filter((r) =>
      `${r.id} ${r.model.id} ${modelName(r.model)} ${instructionLabel(r.model)}`
        .toLowerCase()
        .includes(query.toLowerCase()),
    );
  return (
    <div className="max-w-[1440px] mx-auto px-4 sm:px-8 lg:px-12">
      <header className="min-h-16 py-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-b">
        <a href="/" className="text-2xl font-bold tracking-tight">
          S1MB<span className="text-primary">.</span>
        </a>
        <span className="hidden sm:inline text-xs tracking-widest text-muted-foreground">
          SYSTEM ONE MOSAIC BENCHMARK
        </span>
        <span className="sm:ml-auto text-xs text-muted-foreground">{benchmarks.length} benchmarks · {runs.length} runs</span>
      </header>
      <main className="py-6 space-y-6">
        <h1 className="sr-only">System One evaluations</h1>
        {category.description.includes('Synthetic') && <Badge variant="secondary">Synthetic preview · Not measured</Badge>}
        <Tabs
          value={view}
          onValueChange={(value) => {
            setView(value as View);
            setQuery("");
          }}
        >
          <TabsList variant="line" className="w-full sm:w-auto justify-start">
            <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
            <TabsTrigger value="benchmarks">Benchmarks</TabsTrigger>
            <TabsTrigger value="compare">
              <ArrowLeftRight className="size-4" /> Compare
              {selectedRuns.length > 0 && (
                <Badge variant="secondary" className="ml-1">
                  {selectedRuns.length}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>
        </Tabs>
        {view === "benchmarks" && (
          <div className="flex flex-wrap items-center justify-between gap-4">
            <Tabs value={task} onValueChange={chooseTask}>
              <TabsList variant="line">
                {TASKS.map((t) => (
                  <TabsTrigger key={t} value={t}>
                    {title(t)}{" "}
                    <span className="text-muted-foreground text-xs">
                      {benchmarks.filter((b) => b.task === t).length}
                    </span>
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
            <Button
              variant="outline"
              disabled={!selectedRuns.length}
              onClick={() => {
                setView("compare");
                setQuery("");
              }}
            >
              <ArrowLeftRight className="size-4" /> Compare selected (
              {selectedRuns.length})
            </Button>
          </div>
        )}
        {view === "leaderboard" && (
          <section aria-label="Leaderboard" className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div><h2 className="text-xl font-semibold">Overall leaderboard</h2>
                <p className="text-sm text-muted-foreground mt-1">Baseline-adjusted scores · Higher is better · Equal weight per task</p></div>
              <Button variant="outline" disabled={!selectedRuns.length} onClick={() => { setView('compare'); setQuery(''); }}>
                <ArrowLeftRight className="size-4" /> Compare selected ({selectedRuns.length})
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">Noul, Choice and Score include all benchmarks. General columns show the diverse/contextual subset; they receive no additional ranking weight.</p>
            {allRows.some(r => r.score === null || r.demo) && <label className="flex items-center gap-2 text-xs">
              <Checkbox checked={includeIncomplete} onCheckedChange={v => setIncomplete(v === true)} />Show incomplete & demo runs
            </label>}
            {rows.length ? <div className="rounded-lg border bg-card overflow-hidden">
              <Table aria-label="Overall leaderboard" className="table-fixed text-[9px] sm:text-xs [&_th]:px-0.5 sm:[&_th]:px-1 [&_th]:whitespace-normal [&_th]:break-words [&_td]:px-0.5 sm:[&_td]:px-1 [&_td]:py-2 [&_button]:text-[10px] sm:[&_button]:text-xs">
                <TableHeader><TableRow>
                  <TableHead className="w-6 sm:w-7"><span className="sr-only">Compare</span></TableHead>
                  <TableHead className="w-5 sm:w-8"><span className="sm:hidden" aria-label="Rank">#</span><span className="hidden sm:inline">Rank</span></TableHead><TableHead className="w-[16%] sm:w-[22%]">Model</TableHead>
                  <TableHead className="text-right">Overall</TableHead>
                  {DISPLAY_TASKS.map(t => <TableHead key={t} className="text-right">{title(t)}</TableHead>)}
                  {DISPLAY_TASKS.map(t => <TableHead key={`general-${t}`} className="text-right"><span className="block text-[9px] sm:text-[10px] text-muted-foreground">General</span>{title(t)}</TableHead>)}
                  <TableHead className="hidden sm:table-cell text-right w-14">Coverage</TableHead>
                </TableRow></TableHeader>
                <TableBody>{rows.map((row,i) => <TableRow key={row.runId} data-state={selectedIds.includes(row.runId) ? 'selected' : undefined}>
                  <TableCell><CheckRun id={row.runId} checked={selectedIds.includes(row.runId)} onToggle={toggle} /></TableCell>
                  <TableCell className="font-mono text-muted-foreground">{row.score !== null && !row.demo ? i + 1 : '—'}</TableCell>
                  <TableCell className="whitespace-normal"><ModelLabel model={row.model} runId={row.runId} distinguish={repeatedName(runs, row.model)} onClick={() => setRun(row.runId)} />{row.score === null && <span className="mt-1 block text-xs text-muted-foreground">Aggregate unavailable · {row.results.filter(r => r.status === 'complete').length}/{category.benchmarks.length} complete</span>}{row.demo && <Badge variant="secondary">Demo</Badge>}</TableCell>
                  <TableCell className="text-right">
                    <span className="font-mono font-semibold">{adjustedScore(row.score)}</span>
                  </TableCell>
                  {DISPLAY_TASKS.map(t => <TableCell key={t} className="text-right font-mono">{adjustedScore(diagnosticMean(snapshot, category, t, row.results, 'baseline_adjusted_score'))}</TableCell>)}
                  {DISPLAY_TASKS.map(t => <TableCell key={`general-${t}`} className="text-right font-mono">{adjustedScore(diagnosticMean(snapshot, generalizationCategory(snapshot, category), t, row.results, 'baseline_adjusted_score'))}</TableCell>)}
                  <TableCell className="hidden sm:table-cell text-right text-xs text-muted-foreground">{row.results.filter(r => r.status === 'complete').length}/{category.benchmarks.length}{row.demo && ' · Demo'}</TableCell>
                </TableRow>)}</TableBody>
              </Table>
            </div> : <Empty title="No results yet" description="Measured model results will appear here." />}
            {generalizationColumns.length > 0 && rows.length > 0 && <section aria-label="Generalization benchmark comparison" className="space-y-3 pt-4">
              <h3 className="text-lg font-semibold">Generalization · {generalizationColumns.length} benchmarks</h3>
              <p className="text-xs text-muted-foreground">Adjusted score · 0–100 · Higher is better. 0 means at or below baseline, not zero correct answers. These benchmarks do not establish generalization to unseen tasks or distributions.</p>
              <div className="rounded-lg border bg-card overflow-hidden">
                <Table aria-label="Generalization scores" className="min-w-[700px]">
                  <TableHeader><TableRow>
                    <TableHead>Model</TableHead>
                    {generalizationColumns.map(({ family, task, benchmark }) => <TableHead key={benchmark.id} className="text-right">
                      <span className="block text-xs text-muted-foreground">{title(family)}</span>{title(task)}
                    </TableHead>)}
                  </TableRow></TableHeader>
                  <TableBody>{rows.map(row => <TableRow key={row.runId}>
                    <TableCell className="whitespace-normal"><ModelLabel model={row.model} runId={row.runId} distinguish={repeatedName(runs, row.model)} onClick={() => setRun(row.runId)} />{row.demo && <Badge variant="secondary">Demo</Badge>}</TableCell>
                    {generalizationColumns.map(({ task, benchmark }) => {
                      const value = diagnosticMean(snapshot, { ...category, benchmarks: [benchmark.id] }, task, row.results, 'baseline_adjusted_score');
                      return <TableCell key={benchmark.id} className="text-right font-mono">
                        <button className="underline decoration-muted-foreground/40 underline-offset-4 hover:decoration-current" aria-label={`${modelName(row.model)} · ${benchmark.id}: ${adjustedScore(value)}`} onClick={() => { chooseTask(task); setQuery(''); setBenchmark(benchmark.id); setView('benchmarks'); }}>{adjustedScore(value)}</button>
                      </TableCell>;
                    })}
                  </TableRow>)}</TableBody>
                </Table>
              </div>
              <p className="text-xs text-muted-foreground">Select a score for benchmark details. — means unavailable. The General columns above average Diverse and Contextual within each task.</p>
            </section>}
            <details className="text-xs text-muted-foreground py-2">
              <summary>How scores are calculated</summary>
              <div className="mt-3 space-y-2 leading-relaxed">
                <p>These are baseline-adjusted scores, not accuracy. 0 means at or below the constant baseline; 100 means the reference ceiling.</p>
                <p>Choice: (target mass − baseline) / (1 − baseline). Noul: 2 × balanced accuracy − 1. Score: 1 − MAE / constant baseline MAE.</p>
                <p>Clip each benchmark to 0–100, average benchmarks within each task, then average the three tasks equally. Complete measured coverage is required. Baselines are fitted to evaluation targets.</p>
              </div>
            </details>
          </section>
        )}
        {view === "benchmarks" && (
          <section aria-label="Benchmark results" className="space-y-5">
            <div className="grid sm:grid-cols-[1fr_2fr_auto] items-center gap-3">
              <Input
                aria-label="Search benchmarks"
                placeholder="Search benchmarks…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <Select value={benchmark?.id ?? ""} onValueChange={setBenchmark}>
                <SelectTrigger
                  aria-label="Benchmark"
                  className="w-full bg-card"
                >
                  <SelectValue placeholder="No matching benchmarks" />
                </SelectTrigger>
                <SelectContent>
                  {filteredBenchmarks.map((b) => (
                    <SelectItem key={b.id} value={b.id}>
                      {benchmarkName(b)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="flex items-center gap-1">
                <Button variant="outline" size="icon" aria-label="Previous benchmark" disabled={!benchmark || filteredBenchmarks.indexOf(benchmark) <= 0} onClick={() => setBenchmark(filteredBenchmarks[filteredBenchmarks.indexOf(benchmark!) - 1].id)}><ChevronLeft className="size-4" /></Button>
                <span className="text-xs text-muted-foreground min-w-12 text-center">{benchmark ? filteredBenchmarks.indexOf(benchmark) + 1 : 0} / {filteredBenchmarks.length}</span>
                <Button variant="outline" size="icon" aria-label="Next benchmark" disabled={!benchmark || filteredBenchmarks.indexOf(benchmark) >= filteredBenchmarks.length - 1} onClick={() => setBenchmark(filteredBenchmarks[filteredBenchmarks.indexOf(benchmark!) + 1].id)}><ChevronRight className="size-4" /></Button>
              </div>
            </div>
            {benchmark && individual ? (
              <>
                <div className="flex flex-wrap items-baseline justify-between gap-3">
                  <h2 className="text-2xl font-semibold capitalize">
                    {benchmarkName(benchmark)}
                  </h2>

                  {snapshot.scoring?.[benchmark.id]?.reason && <p className="mt-3 text-sm text-muted-foreground">Excluded from adjusted index: {snapshot.scoring[benchmark.id].reason}</p>}
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary">
                      {benchmark.case_count} test cases
                    </Badge>
                    <Badge variant="secondary">
                      {benchmark.decision_count.toLocaleString()} decisions
                    </Badge>
                    <Badge variant="outline">
                      {METRICS[task].label} · {direction(task)}
                    </Badge>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground" role="status">{individualCells.length} measured results · Ranked by {sortLabel}</p>
                <div className="rounded-lg border bg-card overflow-hidden">
                  <Table aria-label="Benchmark run results" className="min-w-[850px]">
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-12">
                          <span className="sr-only">Compare</span>
                        </TableHead>
                        <TableHead>Model / run</TableHead>
                        {metricColumns.map(m => <TableHead key={m.name} className="text-right" aria-sort={benchmarkMetric === m.name ? metricDirection : 'none'}>
                          <button className={cn('rounded px-1 py-2 underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-ring', benchmarkMetric === m.name && 'text-primary font-bold underline')} onClick={() => setBenchmarkMetric(m.name)} aria-label={`Rank by ${m.label}`}>{m.label}</button>
                        </TableHead>)}
                        <TableHead>Details</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {individualCells.map((cell) => {
                        const r = cell.result!;
                        return (
                          <TableRow
                            key={cell.runId}
                            data-state={
                              selectedIds.includes(cell.runId)
                                ? "selected"
                                : undefined
                            }
                          >
                            <TableCell>
                              <CheckRun
                                id={cell.runId}
                                checked={selectedIds.includes(cell.runId)}
                                onToggle={toggle}
                              />
                            </TableCell>
                            <TableCell className="w-72 min-w-56 max-w-80 py-4 pr-5">
                              <ModelLabel
                                model={r.model}
                                distinguish={repeatedName(runs, r.model)}
                                runId={cell.runId}
                                onClick={() => setRun(cell.runId)}
                              />
                            </TableCell>
                            {metricColumns.map(m => <TableCell key={m.name} className={cn('text-right font-mono', m.name === benchmarkMetric && 'bg-accent/30', m.name === benchmarkMetric && bestMetric != null && r.status === 'complete' && r.provenance === 'measured' && r.metrics[m.name] === bestMetric && 'text-primary font-bold')}>
                              {metricValue(m.name, r.metrics[m.name])}
                            </TableCell>)}
                            <TableCell>{r.status === 'partial' && <Badge variant="secondary">Partial · {r.counts.failed} failed</Badge>}{r.provenance === 'demo' && <Badge variant="secondary">Demo</Badge>}<DiagnosticDetails result={r} /></TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                  {!individualCells.length && (
                    <p className="p-6 text-muted-foreground text-sm">
                      No runs have results for this benchmark.
                    </p>
                  )}
                </div>
                <MetricGuide task={task} result={individualCells.find(c => c.result?.status === 'complete' && c.result.provenance === 'measured')?.result} />
              </>
            ) : (
              <Empty
                title="No matching benchmarks"
                description="Try a different search or task."
              />
            )}
          </section>
        )}
        {view === "compare" && (
          <section aria-label="Compare runs" className="space-y-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold">Compare selected runs</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  Choose exactly which results to put side by side across all
                  tasks.
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={!selectedIds.length}
                onClick={() => setSelected([])}
              >
                Clear selection
              </Button>
            </div>
            <details className="rounded-lg border bg-card p-3" open={selectedRuns.length === 0 || undefined}>
              <summary className="font-medium text-sm">Choose models ({selectedRuns.length} selected)</summary>
              <div className="flex flex-col sm:flex-row gap-4 sm:items-center my-3">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" />
                  <Input
                    className="pl-9"
                    aria-label="Search runs"
                    placeholder="Search models, instructions or run IDs…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
                <label className="flex items-center gap-2 text-xs">
                  <Checkbox
                    checked={includeIncomplete}
                    onCheckedChange={(v) => setIncomplete(v === true)}
                  />
                  Show incomplete & demo runs
                </label>
              </div>
              <div
                className="max-h-64 overflow-y-auto grid md:grid-cols-2 lg:grid-cols-3 gap-2"
                aria-label="Run selection"
              >
                {visibleRuns.map((run) => (
                  <label
                    key={run.id}
                    className={cn(
                      "flex items-start gap-3 rounded-md border p-3 cursor-pointer",
                      selectedIds.includes(run.id) &&
                        "border-primary/50 bg-accent/50",
                    )}
                  >
                    <Checkbox
                      className="mt-1 shrink-0"
                      aria-label={`Select ${run.id}`}
                      checked={selectedIds.includes(run.id)}
                      onCheckedChange={() => toggle(run.id)}
                    />
                    <div className="min-w-0">
                      <ModelLabel model={run.model} runId={run.id} />
                      <p className="text-[11px] text-muted-foreground mt-2">
                        {
                          run.results.filter((r) => r.status === "complete")
                            .length
                        }
                        /{benchmarks.length} complete benchmarks
                        {run.demo ? " · Demo" : ""}
                      </p>
                    </div>
                  </label>
                ))}
              </div>
              {!visibleRuns.length && (
                <p className="text-sm text-muted-foreground">
                  No matching runs.
                </p>
              )}
            </details>
            <ComparisonTables
              snapshot={snapshot}
              categoryId={category.id}
              selectedIds={selectedIds}
              onRemove={toggle}
              onBenchmarkSelect={(id, selectedTask) => { setTask(selectedTask); setBenchmark(id); setBenchmarkMetric('baseline_adjusted_score'); setQuery(''); setRun(''); setView('benchmarks'); window.scrollTo({ top: 0 }); }}
            />
          </section>
        )}
        {active && (
          <RunDetails
            key={active.id}
            run={active}
            benchmarks={benchmarks}
            onClose={() => setRun("")}
          />
        )}
        {snapshot.issues.length > 0 && (
          <section
            role="status"
            className="rounded-lg border border-destructive/40 p-5"
          >
            <h2 className="font-semibold">Some results need attention</h2>
            <ul className="mt-2 list-disc pl-5 text-sm break-words">
              {snapshot.issues.map((issue, i) => (
                <li key={i}>{issue}</li>
              ))}
            </ul>
          </section>
        )}
        <Separator />
        <p className="text-xs text-muted-foreground">S1MB approximates System One capabilities across specialized benchmarks. Scores do not directly measure generalization.</p>
      </main>
      <footer className="border-t py-6 flex flex-wrap justify-between gap-2 text-xs text-muted-foreground">
        <span>S1MB · System One Mosaic Benchmark</span>

      </footer>
    </div>
  );
}
function RunDetails({
  run,
  benchmarks,
  onClose,
}: {
  run: Run;
  benchmarks: Snapshot["benchmarks"];
  onClose: () => void;
}) {
  const returnFocus = useRef<HTMLElement | null>(null);
  const [filter, setFilter] = useState('');
  const [task, setTask] = useState('all');
  const visible = benchmarks.filter(b => (task === 'all' || b.task === task) && `${b.id} ${benchmarkName(b)}`.toLowerCase().includes(filter.trim().toLowerCase()));
  return (
    <Dialog.Root open onOpenChange={open => { if (!open) onClose(); }}>
    <Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-40 bg-black/40" />
      <Dialog.Content
        id="run-details"
        className="fixed z-50 inset-y-3 right-3 left-3 md:left-auto md:w-[min(960px,calc(100vw-24px))] rounded-xl border bg-card shadow-xl overflow-y-auto p-4 sm:p-6 space-y-5"
        onOpenAutoFocus={() => { returnFocus.current = document.activeElement as HTMLElement; }}
        onCloseAutoFocus={event => { event.preventDefault(); returnFocus.current?.focus(); }}
      >
      <Dialog.Title className="sr-only">Run details: {modelName(run.model)}</Dialog.Title>
      <Dialog.Description className="sr-only">Model identity, effective settings and benchmark results.</Dialog.Description>
      <div className="flex justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-widest text-muted-foreground mb-2">
            Run details · {run.demo ? "Demo" : "Self-reported"}
          </p>
          <ModelLabel model={run.model} runId={run.id} detailed />
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          aria-label="Close run details"
        >
          <X className="size-4" />
        </Button>
      </div>
      <details className="text-xs text-muted-foreground">
        <summary>Model identity</summary>
        <dl className="mt-2 space-y-1 break-all"><div>Model: {run.model.id}</div><div>Adapter: {run.model.adapter}</div><div>Revision: {run.model.revision}</div><div>Total params: {run.model.total_params?.toLocaleString('en-US') ?? 'Unknown'}</div><div>Active params: {run.model.active_params?.toLocaleString('en-US') ?? 'Unknown'}</div><div>AP definition: parameters excluding lookup-only embeddings; shared output weights are retained.</div></dl>
      </details>
      <details className="text-sm">
        <summary>Effective model settings</summary>
        <pre className="text-xs p-4 bg-muted rounded mt-3 overflow-auto max-h-80">
          {JSON.stringify(run.model.settings, null, 2)}
        </pre>
      </details>
      <div className="flex flex-wrap gap-3">
        <Input aria-label="Filter model benchmarks" placeholder="Find a benchmark…" value={filter} onChange={event => setFilter(event.target.value)} className="sm:max-w-xs" />
        <Select value={task} onValueChange={setTask}><SelectTrigger aria-label="Model detail task" className="w-36"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All tasks</SelectItem>{TASKS.map(t => <SelectItem key={t} value={t}>{title(t)}</SelectItem>)}</SelectContent></Select>
        <span className="self-center text-xs text-muted-foreground">{visible.length} benchmarks</span>
      </div>
      <Table aria-label="Run benchmark details">
        <TableHeader>
          <TableRow>
            <TableHead>Benchmark</TableHead>
            <TableHead>Task</TableHead>
            <TableHead className="text-right">Adjusted ↑</TableHead>
            <TableHead>Original metric</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {visible.map((b) => (
            <TableRow key={b.id}>
              <TableCell className="whitespace-normal min-w-48">
                <p className="capitalize font-medium">{benchmarkName(b)}</p>
                <p className="font-mono text-[10px] text-muted-foreground break-all">
                  {b.id}
                </p>
              </TableCell>
              <TableCell>
                {title(b.task)} {METRICS[b.task].direction === "up" ? "↑" : "↓"}
              </TableCell>
              <TableCell className="text-right font-mono font-semibold">{percent(run.results.find(r => r.benchmark.id === b.id)?.metrics.baseline_adjusted_score)}</TableCell>
              <TableCell>
                <p className="text-xs text-muted-foreground mb-1">{METRICS[b.task].label}</p>
                <ScoreValue result={run.results.find(r => r.benchmark.id === b.id)} compact />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {!visible.length && <p className="text-sm text-muted-foreground">No matching benchmarks.</p>}
      </Dialog.Content>
    </Dialog.Portal>
    </Dialog.Root>
  );
}
