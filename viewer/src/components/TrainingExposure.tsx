"use client";

import { ChevronDown, GraduationCap, TriangleAlert } from 'lucide-react';
import type { Benchmark, ModelInfo } from '../lib/types';
import { exposuresFor, exposureLabel, exposureRelationship } from '../lib/training-exposure';
import { benchmarkName } from '../lib/comparison';
import { ColumnHelp } from './ColumnHelp';

/** Reported training exposure does not modify scores or prove independence. */
export function TrainingBadge({ model, benchmarkId, benchmarkIds, onClick }: { model: ModelInfo; benchmarkId?: string; benchmarkIds?: string[]; onClick?: () => void }) {
  const entries = exposuresFor(model, benchmarkId).filter(e => !benchmarkIds || benchmarkIds.includes(e.benchmark_id));
  if (!entries.length) return null;
  const label = exposureLabel(entries);
  const Icon = label === 'Known domain' ? GraduationCap : TriangleAlert;
  return <span className="inline-flex align-middle text-primary" data-training-exposure={benchmarkId ?? 'model'}>
    <ColumnHelp label={`${label}: ${benchmarkId ?? model.display_name ?? model.id}`} onClick={onClick}
      description={<div className="max-h-[50vh] space-y-2 overflow-y-auto"><p className="font-semibold">{label} · {new Set(entries.map(e => e.benchmark_id)).size} domains</p><p>{label === 'Known domain' ? 'Reported training used these source families or related domains. This does not establish that evaluation examples were used.' : 'The submitter reports that evaluation examples were used for training.'}</p>{entries.slice(0, 5).map((e, i) => <p key={i}>{e.benchmark_id} · {exposureRelationship[e.relationship]}<br />{e.note}</p>)}{entries.length > 5 && <p className="mt-2">{entries.length - 5} more disclosures in model details.</p>}<p className="mt-2">Submitter-declared. Missing disclosure means unknown. Scores are unchanged.</p></div>}>
      <Icon aria-hidden="true" className="size-3.5 shrink-0" />
    </ColumnHelp>
  </span>;
}

export function TrainingDetails({ model, benchmarks }: { model: ModelInfo; benchmarks: Benchmark[] }) {
  const entries = exposuresFor(model).filter(e => benchmarks.some(b => b.id === e.benchmark_id));
  const domainIds = [...new Set(entries.map(e => e.benchmark_id))];
  return <section aria-label="Training exposure" className="border-t pt-5">
    {!entries.length ? <p className="text-sm text-muted-foreground">Training exposure not disclosed for this scope.</p> : <details className="group rounded-lg border border-primary/30 bg-primary/5 p-4 open:space-y-4">
      <summary className="flex cursor-pointer list-none items-center gap-3 rounded-sm focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-4 [&::-webkit-details-marker]:hidden">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary"><GraduationCap aria-hidden="true" className="size-6" /></span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2 font-semibold"><span>Known domains <span className="sr-only">({domainIds.length})</span></span><span aria-hidden="true" className="rounded-full bg-primary/15 px-2 py-0.5 text-xs text-primary">{domainIds.length}</span></span>
          <span className="mt-1 block text-xs text-muted-foreground">Explore training domains and evidence</span>
        </span>
        <ChevronDown aria-hidden="true" className="size-4 shrink-0 text-primary group-open:rotate-180" />
      </summary>
      <p className="text-xs text-muted-foreground">Submitter-declared training provenance. Same source family or related domain does not mean evaluation examples were used. Missing disclosure means unknown; scores and rankings are unchanged.</p>
      <ul className="space-y-2">
        {domainIds.map(id => {
          const benchmark = benchmarks.find(b => b.id === id)!;
          const disclosures = entries.filter(e => e.benchmark_id === id);
          return <li key={id} className="rounded-md border bg-muted/20 p-3 text-xs [overflow-wrap:anywhere]">
            <details className="space-y-2">
              <summary className="cursor-pointer font-medium">{benchmarkName(benchmark)} · {benchmark.task === 'noul' ? 'Noul' : benchmark.task === 'choice' ? 'Choice' : 'Score'}</summary>
              <p className="font-mono text-muted-foreground">{id}</p>
              {disclosures.map((e, i) => <div key={i} className="space-y-1">
                <p className="font-semibold">{exposureRelationship[e.relationship]}</p>
                <p>{e.note}</p>
                <p className="text-muted-foreground">{e.dataset} · {e.config} · {e.split}</p>
                <p className="font-mono text-muted-foreground">Dataset revision: {e.revision}</p>
                <a className="underline underline-offset-4" href={e.evidence_url} target="_blank" rel="noreferrer">Training evidence ↗</a>
              </div>)}
            </details>
          </li>;
        })}
      </ul>
    </details>}
  </section>;
}
