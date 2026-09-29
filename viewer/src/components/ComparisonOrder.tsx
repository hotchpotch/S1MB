"use client";

import { useState } from 'react';
import { ArrowDown, ArrowUp } from 'lucide-react';
import { categoryRuns, modelName } from '../lib/comparison';
import { diagnosticMean, DISPLAY_TASKS, generalizationCategory, overallIndex, type Category, type Snapshot } from '../lib/types';
import { TASK_AVG_DESCRIPTION } from '../lib/borda';
import { moveComparison } from '../lib/radar';
import { Button } from './ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { SortIcon } from './MetricIcons';
import { ColumnHelp } from './ColumnHelp';
import { ModelName } from './ModelName';

const columns = ['Task Avg', 'Noul', 'Choice', 'Score', 'General Noul', 'General Choice', 'General Score'];

export function ComparisonOrder({ snapshot, category, selectedIds, onChange }: {
  snapshot: Snapshot; category: Category; selectedIds: string[]; onChange: (ids: string[]) => void;
}) {
  const [sort, setSort] = useState<{ column: number; ascending: boolean; ids: string[] } | null>(null);
  const runs = categoryRuns(snapshot, category);
  const general = generalizationCategory(snapshot, category);
  const rows = selectedIds.flatMap(id => {
    const run = runs.find(run => run.id === id);
    return run ? [{ ...run, values: [overallIndex(snapshot, category, run.results),
      ...[category, general].flatMap(group => DISPLAY_TASKS.map(task => diagnosticMean(snapshot, group, task, run.results, 'baseline_adjusted_score')))] }] : [];
  });
  const activeSort = sort && sort.ids.length === selectedIds.length && sort.ids.every((id, index) => id === selectedIds[index]) ? sort : null;
  function sortBy(column: number) {
    const ascending = activeSort?.column === column ? !activeSort.ascending : false;
    const ids = [...rows].sort((a, b) => {
      const av = a.values[column], bv = b.values[column];
      return Number(av == null) - Number(bv == null) || (av != null && bv != null ? (av - bv) * (ascending ? 1 : -1) : 0);
    }).map(row => row.id);
    setSort({ column, ascending, ids });
    onChange(ids);
  }
  if (!rows.length) return null;
  return <section aria-label="Model comparison order" className="space-y-2">
    <h3 className="text-sm font-semibold">Selected models · Display order</h3>
    <p className="text-xs text-muted-foreground">Use the arrows or click a score heading to reorder models and comparison columns. Scores: 0–100, higher is better; — means incomplete or unavailable.</p>
    <div className="rounded-lg border bg-card overflow-hidden">
      <Table aria-label="Selected model order" containerLabel="Scrollable selected models">
        <TableHeader><TableRow>
          <TableHead>Order</TableHead><TableHead>Model</TableHead>
          {columns.map((label, column) => {
            const active = activeSort?.column === column;
            return <TableHead key={label} className="text-right" data-task={column ? DISPLAY_TASKS[(column - 1) % 3] : undefined} aria-sort={active ? activeSort.ascending ? 'ascending' : 'descending' : 'none'}>
              <ColumnHelp onClick={() => sortBy(column)} description={column === 0 ? TASK_AVG_DESCRIPTION : `Sort models by ${label}. Baseline-adjusted scores require complete coverage; unavailable values stay last. Click again to reverse the order.`}>
                {label.startsWith('General ') && <span className="block text-[10px] text-muted-foreground">General</span>}
                <span className="inline-flex items-center whitespace-nowrap">{label.replace('General ', '')}<SortIcon direction={active ? activeSort.ascending ? 'ascending' : 'descending' : 'none'} /></span>
              </ColumnHelp>
            </TableHead>;
          })}
        </TableRow></TableHeader>
        <TableBody>{rows.map((run, index) => <TableRow key={run.id}>
          <TableCell><div className="flex items-center gap-1">
            <span className="w-4 text-muted-foreground tabular-nums">{index + 1}</span>
            {([-1, 1] as const).map(direction => <Button key={direction} variant="ghost" size="icon" className="size-8" aria-label={`Move ${modelName(run.model)} ${direction === -1 ? 'up' : 'down'}`} disabled={direction === -1 ? index === 0 : index === rows.length - 1} onClick={() => {
              setSort(null); onChange(moveComparison(rows.map(row => row.id), run.id, direction));
            }}>{direction === -1 ? <ArrowUp className="size-4" /> : <ArrowDown className="size-4" />}</Button>)}
          </div></TableCell>
          <TableCell className="whitespace-normal min-w-48 max-w-80"><ModelName model={run.model} /></TableCell>
          {run.values.map((value, column) => <TableCell key={column} className={`text-right font-mono tabular-nums ${column === 0 ? 'font-semibold text-primary' : ''}`}>{value == null ? '—' : (value * 100).toFixed(2)}</TableCell>)}
        </TableRow>)}</TableBody>
      </Table>
    </div>
  </section>;
}
