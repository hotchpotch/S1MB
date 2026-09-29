import { ColumnHelp } from './ColumnHelp';
import { generalizationComparison } from '../lib/generalization';
import { modelName } from '../lib/comparison';
import type { Category, Snapshot } from '../lib/types';
import { ParameterCounts, ParameterCountsHeader } from './ParameterCounts';
import { ModelName } from './ModelName';
import { Checkbox } from './ui/checkbox';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';

const score = (value: number | null) => value == null ? '—' : (value * 100).toFixed(2);
const title = (value: string) => value[0].toUpperCase() + value.slice(1);

export function GeneralizationTable({ snapshot, category, selectedIds, onToggle }: {
  snapshot: Snapshot; category: Category; selectedIds: string[];
  onToggle: (id: string) => void;
}) {
  const { columns, rows } = generalizationComparison(snapshot, category);
  return <section aria-label="Generalization benchmark comparison" className="space-y-4">
    <div><h2 className="text-xl font-semibold">Generalization · 6 benchmarks</h2>
      <p className="mt-1 text-sm text-muted-foreground">Diverse and Contextual, each covering Noul, Choice and Score. All six adjusted scores are shown together.</p></div>
    <p className="text-xs text-muted-foreground">0–100 · Higher is better. Average weights the six benchmarks equally and requires all six to be complete and scored. — means incomplete or unavailable; it does not count as zero.</p>
    {rows.length ? <div className="rounded-lg border bg-card overflow-hidden">
      <Table aria-label="Generalization scores" className="table-fixed min-w-[760px] text-xs [&_th]:px-0.5 sm:[&_th]:px-1 [&_th]:whitespace-normal [&_th]:break-words [&_td]:px-0.5 sm:[&_td]:px-1 [&_td]:py-2 [&_tr>:first-child]:pl-3 sm:[&_tr>:first-child]:pl-4 [&_tr>:last-child]:pr-3 sm:[&_tr>:last-child]:pr-4 [&_button]:text-xs">
        <TableHeader><TableRow>
          <TableHead className="w-9 sm:w-11"><span className="sr-only">Compare</span></TableHead>
          <TableHead className="w-[22%]"><ColumnHelp description="Evaluated model. Model links open the authored model page.">Model</ColumnHelp></TableHead>
          <TableHead className="text-right"><ColumnHelp description="Equal-weight average of these six baseline-adjusted benchmark scores. 0–100; higher is better. All six must be complete and scored.">Average ↑</ColumnHelp></TableHead>
          {columns.map(({ family, task, benchmark }) => <TableHead key={benchmark.id} data-task={task} className="task-accent text-right"><ColumnHelp description={`${title(family)} ${title(task)} benchmark, baseline-adjusted on a 0–100 scale. Higher is better; 0 means at or below the reference baseline. — means incomplete or unavailable.`}><span className="block text-[9px] sm:text-[10px] text-muted-foreground">{title(family)}</span>{title(task)}</ColumnHelp></TableHead>)}
          <TableHead className="w-14 sm:w-20 text-right"><ParameterCountsHeader /></TableHead>
        </TableRow></TableHeader>
        <TableBody>{rows.map(row => <TableRow key={row.id} data-state={selectedIds.includes(row.id) ? 'selected' : undefined}>
          <TableCell><Checkbox aria-label={`Compare ${modelName(row.model)}`} checked={selectedIds.includes(row.id)} onCheckedChange={() => onToggle(row.id)} /></TableCell>
          <TableCell className="whitespace-normal [overflow-wrap:anywhere]"><ModelName model={row.model} />{row.demo && <span className="block text-muted-foreground">Demo</span>}</TableCell>
          <TableCell className="text-right font-mono font-semibold tabular-nums">{score(row.average)}</TableCell>
          {columns.map(({ benchmark }, i) => <TableCell key={benchmark.id} className="text-right font-mono tabular-nums">
            {score(row.values[i])}
          </TableCell>)}
          <TableCell><ParameterCounts model={row.model} /></TableCell>
        </TableRow>)}</TableBody>
      </Table>
    </div> : <p className="text-sm text-muted-foreground">No results for these six benchmarks yet.</p>}
    <section aria-label="Understanding zero scores" className="rounded-lg border bg-muted/30 p-4 space-y-2 text-sm">
      <h3 className="font-semibold">Why can a score be 0.00?</h3>
      <p>These scores measure improvement over a simple reference, not the percentage of correct answers. Matching the reference gives 0; doing worse also displays 0 because negative adjusted scores are clipped. Very small positive scores can round to 0.00.</p>
      <p className="text-muted-foreground">For example, if 90% of Noul targets are false, always answering false gets 90% accuracy. But it misses every true case: balanced accuracy is only 50%, so its adjusted score is 0.00. Zero does not mean every answer was wrong.</p>
      <details className="pt-1"><summary className="cursor-pointer font-medium">What is the reference for each task?</summary>
        <ul className="mt-2 list-disc pl-5 space-y-1 text-muted-foreground">
          <li>Choice: the best reference among uniform guessing, always selecting one option ID, or always selecting one option position.</li>
          <li>Noul: 50% balanced accuracy. True and false cases contribute equally; always predicting one class scores zero when both classes are present.</li>
          <li>Score: always predicting the median normalized target value. The adjusted score measures how much the model reduces absolute error against this constant.</li>
        </ul>
        <p className="mt-2 text-xs text-muted-foreground">References are computed from evaluation targets. Each benchmark is adjusted and clipped before averaging.</p>
      </details>
    </section>
    <p className="text-xs text-muted-foreground">These six benchmarks are already included in the overall task scores; they add no extra leaderboard weight. “Generalization” names this general-purpose subset and does not establish performance on unseen tasks or training-data non-overlap.</p>
  </section>;
}
