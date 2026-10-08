"use client";

import { MetricDirection, SortIcon } from "./MetricIcons";
import { ColumnHelp } from "./ColumnHelp";
import { GitHubIcon } from "./GitHubIcon";
import { BenchmarkSources } from "./BenchmarkSources";

import { useEffect, useRef, useState } from "react";
import { ArrowLeftRight, ArrowUpRight, BarChart3, Blocks, Check, CircleCheck, ListChecks, Trophy, ChevronLeft, ChevronRight, Search, SlidersHorizontal, ChevronDown, X } from "lucide-react";
import { ModelWebsiteLink } from "./ModelWebsiteLink";
import { hfModelLabel } from '../lib/hf-model-label';
import { ParameterFilter } from './ParameterFilter';
import { ALL_PARAMETERS, matchesParameterRange, rangeActive, type ParameterRange } from '../lib/parameter-filters';
import { ParameterCounts, ParameterCountsHeader } from "./ParameterCounts";
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
  metricRanks,
  benchmarkName,
  categoryRuns,
  compareRuns,
  hasCompleteCoverage,
  compareResultMetric,
  comparisonSpread,
  comparisonDelta,
  modelName,
  type Run,
} from "../lib/comparison";
import { cn } from "../lib/utils";
import { ModelName } from "./ModelName";
import { GeneralizationTable } from "./GeneralizationTable";
import { bordaScores, BORDA_DESCRIPTION, TASK_AVG_DESCRIPTION } from "../lib/borda";
import { TableExports } from "./TableExports";
import { ComparisonOrder } from "./ComparisonOrder";

type LeaderboardSort = 'borda' | 'overall' | Task | `general-${Task}`;
type View = "leaderboard" | "benchmarks" | "compare";
const adjustedScore = (value: number | null | undefined) => value == null ? '—' : (value * 100).toFixed(2);
const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(2)}%`;
const BENCHMARK_METRICS = {
  choice: [{ name: 'target_mass_at_prediction', label: 'Target mass' }],
  noul: [
    { name: 'f1', label: 'F1' },
    { name: 'precision', label: 'Precision' },
    { name: 'positive_recall', label: 'Recall' },
    { name: 'balanced_accuracy', label: 'Balanced acc.' },
    { name: 'binary_brier', label: 'Brier' },
  ],
  score: [
    { name: 'normalized_expected_score_mae', label: 'MAE' },
    { name: 'normalized_score_rmse', label: 'RMSE' },
  ],
};
const title = (s: string) => s[0].toUpperCase() + s.slice(1);
const repeatedName = (runs: Run[], model: ModelInfo) => runs.filter(r => modelName(r.model) === modelName(model)).length > 1;
const direction = (task: Task) =>
  <MetricDirection direction={METRICS[task].direction}>{METRICS[task].direction === "up" ? "Higher is better" : "Lower is better"}</MetricDirection>;

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
  showRepository = false,
}: {
  model: ModelInfo;
  runId: string;
  onClick?: () => void;
  detailed?: boolean;
  distinguish?: boolean;
  showRepository?: boolean;
}) {
  const repository = showRepository ? hfModelLabel(model.hf_url) : null;
  return (
    <div className={cn('min-w-0 w-full text-left whitespace-normal [overflow-wrap:anywhere]', repository ? 'space-y-0' : 'space-y-1')}>
      {onClick ? (
        <button
          onClick={onClick}
          className={cn('font-semibold text-sm text-foreground underline-offset-4 hover:underline text-left break-words', repository && 'block leading-snug')}
        >
          {detailed ? modelName(model) : model.short_name ?? modelName(model)}
        </button>
      ) : (
        <p className="font-semibold text-sm break-words">{modelName(model)}</p>
      )}
      {repository && (onClick ? (
        <button onClick={onClick} className="block text-left text-[10px]! font-normal leading-tight text-muted-foreground underline-offset-2 hover:underline break-words">
          {repository}
        </button>
      ) : <p className="text-[10px] font-normal leading-tight text-muted-foreground">{repository}</p>)}
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
          "font-mono tabular-nums text-xs",
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
}: {
  snapshot: Snapshot;
  categoryId: string;
  selectedIds: string[];
  onBenchmarkSelect?: (id: string, task: Task) => void;
}) {
  const [filter, setFilter] = useState('');
  const [sort, setSort] = useState('gap');
  const [showDelta, setShowDelta] = useState(false);
  const [referenceId, setReference] = useState('');
  const [adjusted, setAdjusted] = useState(true);
  const category = snapshot.categories.find((c) => c.id === categoryId);
  if (!category) return null;
  const comparison = compareRuns(snapshot, category, selectedIds, adjusted);
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
                {adjusted ? <MetricDirection direction="up">Baseline-adjusted score · Higher is better</MetricDirection> : <>{METRICS[group.task].label} · {direction(group.task)}</>}
              </span>
            </div>
            <div className="rounded-lg border bg-card overflow-hidden">
              <Table aria-label={`${title(group.task)} benchmark comparison`} containerClassName="max-h-[65vh] focus-visible:outline-2 focus-visible:outline-ring" containerLabel={`${title(group.task)} comparison scroll area`} className="table-fixed" style={{ minWidth: 192 + comparison.runs.length * 128 }}>
                <TableHeader>
                  <TableRow>
                    <TableHead className="sticky left-0 top-0 z-30 bg-card w-[128px] sm:w-[220px] whitespace-normal">
                      Benchmark
                    </TableHead>
                    <TableHead className="sticky top-0 z-20 bg-card text-right whitespace-normal w-[64px] sm:w-[100px]">Gap {adjusted ? '(pp)' : ''}</TableHead>
                    {comparison.runs.map((run) => (
                      <TableHead
                        key={run.id}
                        className="sticky top-0 z-20 bg-card py-4 px-3 align-top whitespace-normal [overflow-wrap:anywhere]"
                      >
                        <ModelName model={run.model} />
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
  initialGeneralizationOnly = false,
  initialCheckedOnly = false,
  initialModelSearch = "",
  initialFiltersOpen = false,
  initialCompare = [],
  initialBenchmark,
}: {
  snapshot: Snapshot;
  initialCategory?: string;
  initialTask?: string;
  initialRun?: string;
  initialView?: string;
  initialGeneralizationOnly?: boolean;
  initialCheckedOnly?: boolean;
  initialModelSearch?: string;
  initialFiltersOpen?: boolean;
  initialCompare?: string[];
  initialBenchmark?: string;
}) {
  const leaderboardTable = useRef<HTMLTableElement>(null);
  const categoryId = "english-v1";
  const [task, setTask] = useState<Task>(
    TASKS.includes(initialTask as Task) ? (initialTask as Task) : "choice",
  );
  const [generalizationView, setGeneralizationView] = useState(initialTask === 'generalization');
  const [runId, setRun] = useState(initialRun ?? "");
  const [view, setView] = useState<View>(
    ["leaderboard", "benchmarks", "compare"].includes(initialView ?? "")
      ? (initialView as View)
      : "leaderboard",
  );
  const [selectedIds, setSelected] = useState<string[]>([
    ...new Set(initialCompare),
  ]);
  const [generalizationOnly, setGeneralizationOnly] = useState(initialGeneralizationOnly);
  const [checkedOnly, setCheckedOnly] = useState(initialCheckedOnly);
  const [modelSearch, setModelSearch] = useState(initialModelSearch);
  const [filtersOpen, setFiltersOpen] = useState(initialFiltersOpen);
  const [totalRange, setTotalRange] = useState<ParameterRange>([...ALL_PARAMETERS]);
  const [activeRange, setActiveRange] = useState<ParameterRange>([...ALL_PARAMETERS]);
  const parameterFilterCount = Number(rangeActive(totalRange)) + Number(rangeActive(activeRange));
  const resetParameterFilters = () => { setTotalRange([...ALL_PARAMETERS]); setActiveRange([...ALL_PARAMETERS]); };
  const [includeIncomplete, setIncomplete] = useState(false);
  const [modelPickerOpen, setModelPickerOpen] = useState(true);
  const [query, setQuery] = useState("");
  const [leaderboardSort, setLeaderboardSort] = useState<LeaderboardSort>('borda');
  const [leaderboardAscending, setLeaderboardAscending] = useState(false);
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
      "generalOnly",
      "checkedOnly",
      "modelSearch",
    ])
      params.delete(key);
    params.set("category", category.id);
    params.set("task", generalizationView ? "generalization" : task);
    params.set("view", view);
    if (generalizationOnly) params.set("generalOnly", "1");
    if (checkedOnly) params.set("checkedOnly", "1");
    if (modelSearch) params.set("modelSearch", modelSearch);
    if (runId) params.set("run", runId);
    if (benchmarkId) params.set("benchmark", benchmarkId);
    for (const id of selectedIds) params.append("compare", id);
    window.history.replaceState(null, "", `?${params}`);
  }, [category, task, generalizationView, view, runId, selectedIds, benchmarkId, generalizationOnly, checkedOnly, modelSearch]);
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
  const runs = categoryRuns(snapshot, category);
  const selectedRuns = selectedIds.flatMap(id => { const run = runs.find(r => r.id === id); return run ? [run] : []; });
  const active = runs.find((r) => r.id === runId);
  const rankingCategory = generalizationOnly ? generalizationCategory(snapshot, category) : category;
  const rawRows = leaderboard(snapshot, rankingCategory, task);
  const allRows = rawRows.map(row => ({ ...row, score: overallIndex(snapshot, rankingCategory, row.results) }))
    .sort((a, b) => Number(a.demo) - Number(b.demo) || Number(a.score === null) - Number(b.score === null) || (b.score ?? 0) - (a.score ?? 0) || a.runId.localeCompare(b.runId));
  const rows = allRows.filter(row => hasCompleteCoverage(rankingCategory, row.results));
  const borda = bordaScores(snapshot, rankingCategory);
  const leaderboardValue = (row: typeof rows[number]) => leaderboardSort === 'borda' ? borda.get(row.runId) ?? null : leaderboardSort === 'overall' ? row.score
    : diagnosticMean(snapshot, leaderboardSort.startsWith('general-') ? generalizationCategory(snapshot, category) : rankingCategory,
      leaderboardSort.replace('general-', '') as Task, row.results, 'baseline_adjusted_score');
  const leaderboardRanks = metricRanks(rows.map(row => ({ id: row.runId, value: leaderboardValue(row) })));
  rows.sort((a, b) => {
    const av = leaderboardValue(a), bv = leaderboardValue(b);
    return Number(av == null) - Number(bv == null) || (av != null && bv != null ? (av - bv) * (leaderboardAscending ? 1 : -1) : 0) || a.runId.localeCompare(b.runId);
  });
  const displayedRows = rows.filter(row =>
    (!checkedOnly || selectedIds.includes(row.runId)) &&
    matchesParameterRange(row.model.total_params, totalRange) &&
    matchesParameterRange(row.model.active_params, activeRange) &&
    [row.model.id, row.model.short_name, modelName(row.model)].some(name => name?.toLowerCase().includes(modelSearch.trim().toLowerCase())));
  function sortLeaderboard(column: LeaderboardSort) {
    setLeaderboardAscending(column === leaderboardSort ? !leaderboardAscending : false);
    setLeaderboardSort(column);
  }
  const sortIcon = (column: LeaderboardSort) => <SortIcon direction={sortDirection(column)} />;
  const sortDirection = (column: LeaderboardSort) => column === leaderboardSort ? leaderboardAscending ? 'ascending' as const : 'descending' as const : 'none' as const;
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
  const metricColumns = [{ name: 'baseline_adjusted_score', label: 'Adjusted' }, ...BENCHMARK_METRICS[task]];
  const sortLabel = metricColumns.find(m => m.name === benchmarkMetric)?.label ?? 'Adjusted';
  const metricDirection = ['binary_brier', 'normalized_expected_score_mae', 'normalized_score_rmse'].includes(benchmarkMetric) ? 'ascending' : 'descending';
  const benchmarkRanks = metricRanks(individualCells.map(cell => ({ id: cell.runId,
    value: cell.result?.status === 'complete' && cell.result.provenance === 'measured' ? cell.result.metrics[benchmarkMetric] ?? null : null,
  })), metricDirection === 'ascending');
  const bestMetric = individualCells.find(c => c.result?.status === 'complete' && c.result.provenance === 'measured' && c.result.metrics[benchmarkMetric] != null)?.result?.metrics[benchmarkMetric];
  function toggle(id: string) {
    setSelected((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id],
    );
  }
  function chooseTask(value: string) {
    setGeneralizationView(value === 'generalization');
    if (value === 'generalization') { setBenchmark(''); setQuery(''); return; }
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
      `${r.id} ${r.model.id} ${r.model.short_name ?? ''} ${modelName(r.model)} ${instructionLabel(r.model)}`
        .toLowerCase()
        .includes(query.toLowerCase()),
    );
  return (
    <div className="max-w-[1440px] mx-auto px-4 sm:px-8 lg:px-12">
      <main className="pb-6 space-y-6">
        <section aria-label="About S1MB" className="mosaic-hero -mx-4 sm:-mx-8 lg:-mx-12 border-b bg-background text-foreground px-4 sm:px-8 lg:px-12 py-8 sm:py-10">
          <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr] lg:items-end lg:gap-12">
            <div>
              <a href="/" aria-label="S1MB home" className="mb-6 inline-block text-2xl font-bold tracking-tight">S1MB<span className="text-primary">.</span></a>
              <h1 className="text-[clamp(2.8rem,5.1vw,4.6rem)] font-semibold leading-[0.98] tracking-[-0.06em]">System One<br /><span className="text-primary">Mosaic</span> Benchmark</h1>
            </div>
            <div className="w-full max-w-lg space-y-5 lg:justify-self-end lg:pb-1">
              <p className="text-[clamp(1.6rem,2.5vw,2.25rem)] font-semibold leading-[1.08] tracking-[-0.06em] text-foreground">Compare models across<br /><span className="text-primary">{benchmarks.length} specialized benchmarks.</span></p>
              <div className="grid grid-cols-3 gap-3 sm:gap-5">
                {([
                  { task: 'noul', label: 'Noul', description: 'Assess a condition', Icon: CircleCheck },
                  { task: 'choice', label: 'Choice', description: 'Select an option', Icon: ListChecks },
                  { task: 'score', label: 'Score', description: 'Rate on a scale', Icon: BarChart3 },
                ] as const).map(({ task, label, description, Icon }) => <div key={task} className="space-y-2">
                  <div data-task={task} className="task-accent border-t-2 border-current pt-3 flex items-center gap-2 text-sm font-semibold"><Icon aria-hidden="true" className="size-4 shrink-0" strokeWidth={1.5} />{label}</div>
                  <p className="text-[11px] sm:text-xs leading-relaxed text-muted-foreground">{description}</p>
                </div>)}
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] sm:text-xs text-muted-foreground">
                <Badge variant="outline" className="rounded-md border-primary/20 bg-primary/5 font-mono text-[10px] text-primary sm:text-[11px]">{category.id}</Badge>
                <a href="https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark/" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 transition-colors hover:text-primary">
                  <span className="underline underline-offset-2">Explainer article</span>
                  <ArrowUpRight aria-hidden="true" className="size-3" />
                </a>
                <a href="https://github.com/hotchpotch/S1MB" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 transition-colors hover:text-primary">
                  <GitHubIcon />
                  <span className="underline underline-offset-2">hotchpotch/S1MB</span>
                  <ArrowUpRight aria-hidden="true" className="size-3" />
                </a>
              </div>
            </div>
          </div>
        </section>
        {category.description.includes('Synthetic') && <Badge variant="secondary">Synthetic preview · Not measured</Badge>}
        <Tabs
          value={view}
          onValueChange={(value) => {
            setView(value as View);
            setQuery("");
          }}
        >
          <TabsList variant="line" className="w-full sm:w-auto justify-start [&_button]:gap-1 [&_button]:px-1 [&_button]:text-xs sm:[&_button]:gap-1.5 sm:[&_button]:px-2 sm:[&_button]:text-sm">
            <TabsTrigger value="leaderboard"><Trophy aria-hidden="true" className="size-4" />Leaderboard</TabsTrigger>
            <TabsTrigger value="benchmarks"><Blocks aria-hidden="true" className="size-4" />Benchmarks</TabsTrigger>
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
            <Tabs className="w-full min-w-0 sm:w-auto" value={generalizationView ? "generalization" : task} onValueChange={chooseTask}>
              <TabsList variant="line" className="w-full sm:w-auto [&_button]:gap-1 [&_button]:px-1 [&_button]:text-xs sm:[&_button]:gap-1.5 sm:[&_button]:px-2 sm:[&_button]:text-sm">
                {TASKS.map((t) => (
                  <TabsTrigger key={t} value={t}><span aria-hidden="true" data-task={t} className="task-accent size-1.5 rounded-full bg-current" />
                    {title(t)}{" "}
                    <span className="hidden text-muted-foreground text-xs sm:inline">
                      {benchmarks.filter((b) => b.task === t).length}
                    </span>
                  </TabsTrigger>
                ))}
                <TabsTrigger value="generalization">Generalization <span className="hidden text-muted-foreground text-xs sm:inline">6</span></TabsTrigger>
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
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
              <h2 className="text-xl font-semibold">{generalizationOnly ? 'Generalization leaderboard' : 'Overall leaderboard'}</h2>
                <div className="flex flex-wrap items-center gap-2">
                  <label title="Rank only the Diverse and Contextual Noul, Choice and Score benchmarks" className={cn('inline-flex h-8 cursor-pointer items-center gap-2 whitespace-nowrap rounded-md border px-2.5 text-xs transition-colors hover:bg-accent has-focus-visible:ring-2 has-focus-visible:ring-ring', generalizationOnly ? 'border-primary/30 bg-primary/10 text-primary' : 'bg-background')}>
                    <Checkbox className="size-3.5" checked={generalizationOnly} onCheckedChange={checked => {
                      setGeneralizationOnly(checked === true);
                      setLeaderboardSort('borda'); setLeaderboardAscending(false);
                    }} />
                    Generalization tasks only
                  </label>
                  <label title="Show checked models without recalculating scores or ranks" className={cn('inline-flex h-8 cursor-pointer items-center gap-2 whitespace-nowrap rounded-md border px-2.5 text-xs transition-colors hover:bg-accent has-focus-visible:ring-2 has-focus-visible:ring-ring', checkedOnly ? 'border-primary/30 bg-primary/10 text-primary' : 'bg-background')}>
                    <Checkbox className="size-3.5" checked={checkedOnly} onCheckedChange={checked => setCheckedOnly(checked === true)} />
                    Checked only
                  </label>
                  <Button variant="outline" size="sm" className="text-xs" disabled={!selectedRuns.length} onClick={() => { setView('compare'); setQuery(''); }}>
                    <ArrowLeftRight className="size-3.5" /> Compare ({selectedRuns.length})
                  </Button>
                </div>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <div className="flex items-center gap-3 text-xs text-muted-foreground" role="status">
                    <span><strong className="font-mono text-foreground">{displayedRows.length}{displayedRows.length !== rows.length && <span className="font-normal text-muted-foreground"> / {rows.length}</span>}</strong> models</span>
                    <span className="h-3 border-l" aria-hidden="true" />
                    <span><strong className="font-mono text-foreground">{rankingCategory.benchmarks.length}</strong> benchmarks</span>
                  </div>
                  <div className="flex items-center gap-0.5 border-l pl-2">
                    <TableExports tableRef={leaderboardTable} filename={`s1mb-${generalizationOnly ? 'generalization' : 'all'}-${leaderboardSort}-${leaderboardAscending ? 'asc' : 'desc'}`} disabled={!displayedRows.length} />
                  </div>
                  <div className="relative w-44">
                    <Search aria-hidden="true" className="absolute left-2 top-2 size-3.5 text-muted-foreground" />
                    <Input aria-label="Search leaderboard models" placeholder="Search models…" value={modelSearch} onChange={event => setModelSearch(event.target.value)} className="h-8 pl-7 pr-7 text-xs md:text-xs" />
                    {modelSearch && <Button variant="ghost" size="icon-xs" aria-label="Clear model search" className="absolute right-1 top-1" onClick={() => setModelSearch('')}><X className="size-3" /></Button>}
                  </div>
                  <Button variant="outline" size="sm" className={cn('h-8 text-xs', parameterFilterCount > 0 && 'border-primary/30 bg-primary/10 text-primary')} aria-expanded={filtersOpen} aria-controls="leaderboard-filters" onClick={() => setFiltersOpen(open => !open)}>
                    <SlidersHorizontal className="size-3.5" /> Filters
                    {parameterFilterCount > 0 && <Badge className="h-4 min-w-4 px-1 text-[10px]">{parameterFilterCount}</Badge>}
                    <ChevronDown className={cn('size-3.5 transition-transform', filtersOpen && 'rotate-180')} />
                  </Button>
                </div>

              </div>
            </div>
            <div id="leaderboard-filters" inert={!filtersOpen} className={cn('grid transition-[grid-template-rows,opacity] duration-200 motion-reduce:transition-none', filtersOpen ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0')}>
              <div className="overflow-hidden">
                <div className="border-y py-3">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <p className="text-[11px] text-muted-foreground">Parameters</p>
                    <Button variant="ghost" size="sm" className="h-6 px-2 text-[11px] text-muted-foreground" disabled={!parameterFilterCount} onClick={resetParameterFilters}>Reset all</Button>
                  </div>
                  <div className="grid max-w-3xl gap-4 sm:grid-cols-2 sm:gap-8">
                    <ParameterFilter label="TP · Total parameters" description="Total model parameters. M = million; B = billion." value={totalRange} onChange={setTotalRange} />
                    <ParameterFilter label="AP · Active parameters" description="Static AP excludes embedding weights, including shared output embeddings. M = million; B = billion." value={activeRange} onChange={setActiveRange} />
                  </div>
                  <p className="mt-3 text-[11px] text-muted-foreground">Unknown parameter counts are excluded when a range is set.</p>
                </div>
              </div>
            </div>
            {displayedRows.length ? <div className="rounded-lg border bg-card overflow-hidden">
              <Table ref={leaderboardTable} aria-label="Overall leaderboard" className="table-fixed min-w-[840px] text-xs [&_th]:px-0.5 sm:[&_th]:px-1 [&_th]:whitespace-normal [&_th]:break-words [&_td]:px-0.5 sm:[&_td]:px-1 [&_td]:py-2 [&_tr>:first-child]:pl-3 sm:[&_tr>:first-child]:pl-4 [&_tr>:last-child]:pr-3 sm:[&_tr>:last-child]:pr-4 [&_button]:text-xs">
                <TableHeader><TableRow>
                  <TableHead className="w-9 sm:w-11"><span className="sr-only">Compare</span></TableHead>
                  <TableHead className="w-5 sm:w-8"><ColumnHelp label="Rank" description={'Position based on the selected score column.\n\nHigher scores rank first. Tied models share a rank. Reversing the display order does not change ranks.\n\nOnly models with complete measured results are ranked.'}>#</ColumnHelp></TableHead><TableHead className="w-[22%]"><ColumnHelp description={'The evaluated model.\n\nSelect its name to see model details and individual benchmark results.'}>Model</ColumnHelp></TableHead>
                  <TableHead className="text-right" aria-sort={sortDirection('borda')}><ColumnHelp label="Borda Score" onClick={() => sortLeaderboard('borda')} description={`${BORDA_DESCRIPTION}\n\nThis ranking covers ${rankingCategory.benchmarks.length} benchmarks.`}><span className="block">Borda</span><span className="inline-flex items-center whitespace-nowrap">Score{sortIcon('borda')}</span></ColumnHelp></TableHead>
                  <TableHead className="text-right" aria-sort={sortDirection('overall')}><ColumnHelp onClick={() => sortLeaderboard('overall')} label="Task Avg" description={TASK_AVG_DESCRIPTION}><span className="block">Task</span><span className="inline-flex items-center whitespace-nowrap">Avg{sortIcon('overall')}</span></ColumnHelp></TableHead>
                  {!generalizationOnly && DISPLAY_TASKS.map(t => <TableHead key={t} data-task={t} className="task-accent text-right" aria-sort={sortDirection(t)}><ColumnHelp onClick={() => sortLeaderboard(t)} description={`Average performance on all ${title(t)} benchmarks (0–100; higher is better).\n\nEach benchmark is adjusted against its baseline, clipped to 0–100, then given equal weight.\n\n0 = at or below the baseline; 100 = the reference ceiling. This is not accuracy.\n\nIncludes General ${title(t)}. Requires complete results for this task.`}><span className="inline-flex items-center whitespace-nowrap">{title(t)}{sortIcon(t)}</span></ColumnHelp></TableHead>)}
                  {DISPLAY_TASKS.map(t => <TableHead key={`general-${t}`} data-task={t} className="task-accent text-right" aria-sort={sortDirection(`general-${t}`)}><ColumnHelp onClick={() => sortLeaderboard(`general-${t}`)} description={`General ${title(t)} limits the score to synthetic datasets designed to evaluate generalization, rather than all ${title(t)} benchmarks.\n\nIncludes the Diverse and Contextual ${title(t)} benchmarks. Their baseline-adjusted scores receive equal weight (0–100; higher is better). Both must be complete.\n\n${generalizationOnly ? 'Only the General benchmarks contribute to Task Avg and Borda Score in this view.' : `These benchmarks already count in ${title(t)}, Task Avg and Borda Score. This column adds no extra weight.`}\n\nThe evaluation purpose does not establish generalization to unseen tasks or rule out training-data overlap.`}><span className="block text-[9px] sm:text-[10px] text-muted-foreground">General</span><span className="inline-flex items-center whitespace-nowrap">{title(t)}{sortIcon(`general-${t}`)}</span></ColumnHelp></TableHead>)}
                  <TableHead className="hidden sm:table-cell text-right w-14"><ColumnHelp label="Coverage" description={'Number of completed benchmarks in the selected ranking scope.\n\nModels with complete measured coverage of the selected scope or its General subset appear here. Full-scope aggregates remain unavailable until that scope is complete.'}>Cov</ColumnHelp></TableHead>
                  <TableHead data-export-columns="TP,AP" className="w-14 sm:w-20 text-right"><ParameterCountsHeader /></TableHead>
                </TableRow></TableHeader>
                <TableBody>{displayedRows.map((row) => <TableRow key={row.runId} className="cursor-pointer data-[state=selected]:bg-primary/10" data-state={selectedIds.includes(row.runId) ? 'selected' : undefined}
                  onClick={event => {
                    if (event.defaultPrevented || !(event.target instanceof Element)) return;
                    if (event.target.closest('a, button, input, select, textarea, label, summary, [role="button"], [role="checkbox"], [role="link"], [contenteditable="true"]')) return;
                    toggle(row.runId);
                  }}>
                  <TableCell><CheckRun id={row.runId} checked={selectedIds.includes(row.runId)} onToggle={toggle} /></TableCell>
                  <TableCell className="font-mono text-muted-foreground">{leaderboardRanks.get(row.runId) ?? '—'}</TableCell>
                  <TableCell data-export-url={row.model.url} data-export-huggingface={row.model.hf_url} className="whitespace-normal"><ModelLabel showRepository model={row.model} runId={row.runId} distinguish={repeatedName(runs, row.model)} onClick={() => setRun(row.runId)} />{row.score === null && <span className="mt-1 block text-xs text-muted-foreground">Aggregate unavailable · {row.results.filter(r => rankingCategory.benchmarks.includes(r.benchmark.id) && r.status === 'complete').length}/{rankingCategory.benchmarks.length} complete</span>}{row.demo && <Badge variant="secondary">Demo</Badge>}</TableCell>
                  <TableCell className="text-right font-mono font-semibold text-primary">{adjustedScore(borda.get(row.runId))}</TableCell>
                  <TableCell className="text-right">
                    <span className="font-mono">{adjustedScore(row.score)}</span>
                  </TableCell>
                  {!generalizationOnly && DISPLAY_TASKS.map(t => <TableCell key={t} className="text-right font-mono">{adjustedScore(diagnosticMean(snapshot, category, t, row.results, 'baseline_adjusted_score'))}</TableCell>)}
                  {DISPLAY_TASKS.map(t => <TableCell key={`general-${t}`} className="text-right font-mono">{adjustedScore(diagnosticMean(snapshot, generalizationCategory(snapshot, category), t, row.results, 'baseline_adjusted_score'))}</TableCell>)}
                  <TableCell className="hidden sm:table-cell text-right text-xs text-muted-foreground">{row.results.filter(r => rankingCategory.benchmarks.includes(r.benchmark.id) && r.status === 'complete').length}{row.demo && ' · Demo'}</TableCell>
                  <TableCell><ParameterCounts model={row.model} /></TableCell>
                </TableRow>)}</TableBody>
              </Table>
            </div> : rows.length ? <div className="rounded-lg border bg-muted/20 p-6 text-center space-y-3">
              <p className="text-sm font-medium">{checkedOnly && !selectedIds.length ? 'Check models to show them here' : 'No models match these filters'}</p>
              <p className="text-xs text-muted-foreground">Filters do not change Borda scores or ranks.</p>
              <Button variant="outline" size="sm" onClick={() => { setCheckedOnly(false); setModelSearch(''); resetParameterFilters(); }}>Show all models</Button>
            </div> : <Empty title="No complete models yet" description="Only measured models with every benchmark in the selected scope complete appear on the leaderboard. Use Compare to inspect partial results." />}
            <details className="text-xs text-muted-foreground py-2">
              <summary>Scoring & scope</summary>
              <div className="mt-3 space-y-2 leading-relaxed">
                <p>{generalizationOnly ? 'Only the Diverse and Contextual benchmarks contribute to this ranking. Borda, Task Avg and eligibility are recalculated within this subset.' : 'All benchmarks contribute to this ranking. General columns show the Diverse and Contextual subset already included in the task scores, with no extra weight.'} This subset does not establish unseen-task generalization.</p>
                <p className="whitespace-pre-line">{BORDA_DESCRIPTION}</p>
                <p>Task Avg and task columns are baseline-adjusted scores, not accuracy. 0 means at or below the constant baseline; 100 means the reference ceiling.</p>
                <p>Choice: (target mass − baseline) / (1 − baseline). Noul: 2 × balanced accuracy − 1. Score: 1 − MAE / constant baseline MAE.</p>
                <p>Clip each benchmark to 0–100, average benchmarks within each task, then average the three tasks equally. Complete measured coverage is required. Baselines are fitted to evaluation targets.</p>
              </div>
            </details>
          </section>
        )}
        {view === "benchmarks" && generalizationView && <GeneralizationTable snapshot={snapshot} category={category} selectedIds={selectedIds} onToggle={toggle} />}
        {view === "benchmarks" && !generalizationView && (
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
                  <div className="space-y-1.5">
                  <h2 className="text-xl font-semibold capitalize">
                    {benchmarkName(benchmark)}
                  </h2>
                  <BenchmarkSources dataset={benchmark.dataset} />
                  </div>

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
                        <TableHead className="w-12">#</TableHead>
                        <TableHead>Model / run</TableHead>
                        {metricColumns.map(m => <TableHead key={m.name} className="text-right" aria-sort={benchmarkMetric === m.name ? metricDirection : 'none'}>
                          <button className={cn('inline-flex items-center whitespace-nowrap rounded px-1 py-2 underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-ring', benchmarkMetric === m.name && 'text-primary font-bold underline')} onClick={() => setBenchmarkMetric(m.name)} aria-label={`Rank by ${m.label}`}>{m.label}<SortIcon direction={benchmarkMetric === m.name ? metricDirection : 'none'} /></button>
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
                            <TableCell className="font-mono text-muted-foreground">{benchmarkRanks.get(cell.runId) ?? '—'}</TableCell>
                            <TableCell className="w-72 min-w-56 max-w-80 py-2 pr-5">
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
                <h2 className="text-xl font-semibold">Compare models</h2>
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
            <details className="rounded-lg border bg-card p-3" open={modelPickerOpen}>
              <summary className="cursor-pointer font-medium text-sm" onClick={event => { event.preventDefault(); setModelPickerOpen(open => !open); }}>Choose models ({selectedRuns.length} selected)</summary>
              <div className="flex flex-col sm:flex-row gap-4 sm:items-center my-3">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" />
                  <Input
                    className="pl-9"
                    aria-label="Search models"
                    placeholder="Search models…"
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
              {visibleRuns.length > 0 ? <Table aria-label="Choose models" containerClassName="max-h-72">
                <TableHeader><TableRow><TableHead className="w-10"><span className="sr-only">Select</span></TableHead><TableHead>Model</TableHead><TableHead className="text-right">Complete benchmarks</TableHead></TableRow></TableHeader>
                <TableBody>{visibleRuns.map(run => <TableRow key={run.id} className="cursor-pointer" data-state={selectedIds.includes(run.id) ? 'selected' : undefined} onClick={event => {
                  if (event.target instanceof Element && event.target.closest('[role="checkbox"]')) return;
                  toggle(run.id);
                }}>
                  <TableCell><Checkbox aria-label={`Select ${run.model.short_name ?? modelName(run.model)}`} checked={selectedIds.includes(run.id)} onCheckedChange={() => toggle(run.id)} /></TableCell>
                  <TableCell className="whitespace-normal">
                    <span className="block font-semibold text-sm break-words">{run.model.short_name ?? modelName(run.model)}</span>
                    <span className="mt-1 block text-[11px] text-muted-foreground break-words">{modelName(run.model)}</span>
                  </TableCell>
                  <TableCell className="text-right text-xs text-muted-foreground">{run.results.filter(r => r.status === 'complete').length}/{benchmarks.length}{run.demo ? ' · Demo' : ''}</TableCell>
                </TableRow>)}</TableBody>
              </Table> : (
                <div className="rounded-lg border border-dashed bg-muted/20 px-4 py-8 text-center">
                  <div role="status">
                    <Search aria-hidden="true" className="mx-auto mb-3 size-5 text-muted-foreground" />
                    <p className="text-sm font-semibold">No matching runs</p>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">Try another model name or include incomplete and demo runs.</p>
                  </div>
                  {query && <Button variant="outline" size="sm" className="mt-4" onClick={() => setQuery('')}>Clear search</Button>}
                </div>
              )}
            </details>
            <ComparisonOrder snapshot={snapshot} category={category} selectedIds={selectedIds} onChange={setSelected} />
            <ComparisonTables
              snapshot={snapshot}
              categoryId={category.id}
              selectedIds={selectedIds}
              onBenchmarkSelect={(id, selectedTask) => { chooseTask(selectedTask); setBenchmark(id); setBenchmarkMetric('baseline_adjusted_score'); setQuery(''); setRun(''); setView('benchmarks'); window.scrollTo({ top: 0 }); }}
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
        {snapshot.display && (
          <section aria-label="Display data" className="text-xs text-muted-foreground space-y-1 break-all">
            <p>Results prepared: {snapshot.display.generatedAt}</p>
            {snapshot.display.source && <a className="underline" href={`https://huggingface.co/datasets/${snapshot.display.source.repo}/tree/${snapshot.display.source.revision}`} target="_blank" rel="noreferrer">Results source · {snapshot.display.source.revision.slice(0, 12)}</a>}
          </section>
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
      <Dialog.Description className="sr-only">Model information and benchmark results.</Dialog.Description>
      <div className="flex justify-between gap-3">
        <div className="min-w-0 space-y-2">
          <p className="text-xs uppercase tracking-widest text-muted-foreground mb-2">
            Run details · {run.demo ? "Demo" : "Self-reported"}
          </p>
          <h2 className="text-xl font-semibold tracking-tight [overflow-wrap:anywhere]">{modelName(run.model)}</h2>
          <p className="text-xs text-muted-foreground">{instructionLabel(run.model)}</p>
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
      {(run.model.url || run.model.hf_url) && <div className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
        {run.model.url && run.model.url !== run.model.hf_url && <ModelWebsiteLink url={run.model.url} />}
        {run.model.hf_url && <ModelWebsiteLink url={run.model.hf_url} />}
      </div>}
      <section aria-label="Benchmark results" className="space-y-3 border-t pt-5">
      <div className="flex items-baseline justify-between gap-3"><h3 className="font-semibold">Benchmark results</h3><span className="text-xs text-muted-foreground">{visible.length} / {benchmarks.length} benchmarks</span></div>
      <div className="flex flex-wrap gap-2">
        <Input aria-label="Filter model benchmarks" placeholder="Find a benchmark…" value={filter} onChange={event => setFilter(event.target.value)} className="min-w-0 flex-1 basis-44" />
        <Select value={task} onValueChange={setTask}><SelectTrigger aria-label="Model detail task" className="w-36"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All tasks</SelectItem>{TASKS.map(t => <SelectItem key={t} value={t}>{title(t)}</SelectItem>)}</SelectContent></Select>
      </div>
      <div className="rounded-lg border overflow-hidden"><Table aria-label="Run benchmark details" containerClassName="max-h-[50vh]" containerLabel="Benchmark results scroll area" className="[&_td]:align-top [&_td]:py-3 [&_th]:sticky [&_th]:top-0 [&_th]:z-10 [&_th]:bg-card">
        <TableHeader>
          <TableRow>
            <TableHead>Benchmark</TableHead>
            <TableHead>Task</TableHead>
            <TableHead className="text-right"><MetricDirection direction="up">Adjusted</MetricDirection></TableHead>
            <TableHead>Original metric</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {visible.map((b) => (
            <TableRow key={b.id}>
              <TableCell className="whitespace-normal min-w-48">
                <p className="capitalize font-medium text-xs leading-5">{benchmarkName(b)}</p>
                <p className="mt-1 font-mono text-[10px] leading-4 text-muted-foreground break-all">
                  {b.id}
                </p>
              </TableCell>
              <TableCell className="text-xs leading-5 task-accent" data-task={b.task}>
                {title(b.task)}
              </TableCell>
              <TableCell className="text-right text-xs leading-5 font-mono font-semibold tabular-nums">{percent(run.results.find(r => r.benchmark.id === b.id)?.metrics.baseline_adjusted_score)}</TableCell>
              <TableCell>
                <ScoreValue result={run.results.find(r => r.benchmark.id === b.id)} compact />
                <p className="mt-1 text-xs leading-4 text-muted-foreground"><MetricDirection direction={METRICS[b.task].direction}>{METRICS[b.task].label}</MetricDirection></p>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table></div>
      {!visible.length && <p className="text-sm text-muted-foreground">No matching benchmarks.</p>}
      </section>
      <details className="rounded-lg border bg-muted/20 p-4 text-sm">
        <summary className="font-medium">Evaluation details</summary>
        <div className="mt-4 space-y-4">
          <div className="text-xs text-muted-foreground"><span className="font-medium">Run ID</span><p className="mt-1 font-mono break-all">{run.id}</p></div>
      <details className="text-xs text-muted-foreground">
        <summary>Model identity</summary>
        <dl className="mt-2 space-y-1 break-all"><div>Model: {run.model.id}</div><div>Adapter: {run.model.adapter}</div><div>Total params: {run.model.total_params?.toLocaleString('en-US') ?? 'Unknown'}</div><div>Active params: {run.model.active_params?.toLocaleString('en-US') ?? 'Unknown'}</div><div>AP definition: {run.model.parameter_count_method === "non_lookup_parameters_v1" ? "Legacy: excludes lookup-only embeddings; tied output embeddings retained." : "Static: TP minus embedding weights, including tied output embeddings."}</div></dl>
      </details>
        </div>
      </details>
      </Dialog.Content>
    </Dialog.Portal>
    </Dialog.Root>
  );
}
