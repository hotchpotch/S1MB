"use client";

import { ChevronDown, GraduationCap, TriangleAlert } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { Benchmark, ModelInfo } from '../lib/types';
import { exposuresFor, exposureLabel, exposureRelationship } from '../lib/training-exposure';
import { benchmarkName } from '../lib/comparison';
import { ColumnHelp } from './ColumnHelp';

const knownDomainContext = 'Known domain means disclosed training on the same source family or a related domain. It does not establish that S1MB evaluation examples were used or that record-level overlap was independently audited. Exact training revisions and splits may be undisclosed; inspect each entry’s evidence and limitations.';

const disclosureContext = 'This mark makes documented training exposure transparent; it does not explain or predict a model’s score. Other models may have trained on these domains even when their public information does not confirm it. Scores and rankings are unchanged.';

/** Reported training exposure does not modify scores or prove independence. */
export function TrainingBadge({ model, benchmarkId, benchmarkIds, onClick }: { model: ModelInfo; benchmarkId?: string; benchmarkIds?: string[]; onClick?: () => void }) {
  const entries = exposuresFor(model, benchmarkId).filter(e => !benchmarkIds || benchmarkIds.includes(e.benchmark_id));
  if (!entries.length) return null;
  const known = new Set(entries.map(e => e.benchmark_id)).size;
  const total = benchmarkId ? 1 : benchmarkIds ? new Set(benchmarkIds).size : null;
  const coverage = total == null ? `${known} benchmarks` : `${known}/${total} benchmarks in this scope`;
  const label = exposureLabel(entries);
  const Icon = label === 'Known domain' ? GraduationCap : TriangleAlert;
  return <span className="inline-flex align-middle text-primary" data-training-exposure={benchmarkId ?? 'model'}>
    <ColumnHelp label={`${label}: ${coverage} · ${benchmarkId ?? model.display_name ?? model.id}`} onClick={onClick}
      description={<><p className="font-semibold">{label} · {coverage}</p><p>{onClick ? 'Click to view training disclosures.' : 'Disclosed training exposure.'}</p></>}>
      <Icon aria-hidden="true" className="size-3.5 shrink-0" />
    </ColumnHelp>
  </span>;
}

export function TrainingDetails({ model, benchmarks, focus = '' }: { model: ModelInfo; benchmarks: Benchmark[]; focus?: string }) {
  const entries = exposuresFor(model).filter(e => benchmarks.some(b => b.id === e.benchmark_id));
  const domainIds = [...new Set(entries.map(e => e.benchmark_id))];
  const selected = entries.length > 0 && (focus === 'all' || domainIds.includes(focus));
  const [expanded, setExpanded] = useState(selected);
  const panel = useRef<HTMLDetailsElement>(null);
  const target = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (!selected) return;
    setExpanded(true);
    const frame = requestAnimationFrame(() => {
      const element = focus === 'all' ? panel.current : target.current;
      element?.scrollIntoView({ block: 'nearest' });
      element?.focus({ preventScroll: true });
    });
    return () => cancelAnimationFrame(frame);
  }, [focus, selected]);
  return <section aria-label="Training exposure" className="border-t pt-5">
    {!entries.length ? <p className="text-sm text-muted-foreground">Training exposure is unknown / not disclosed for this scope. The model may still have trained on these domains.</p> : <details ref={panel} tabIndex={-1} open={expanded} onToggle={event => setExpanded(event.currentTarget.open)} data-training-focus={selected ? focus : undefined} className={`group rounded-lg border border-primary/30 outline-none bg-primary/5 p-4 open:space-y-4 ${selected ? 'ring-2 ring-primary ring-offset-2 ring-offset-card' : ''}`}>
      <summary className="flex cursor-pointer list-none items-center gap-3 rounded-sm focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-4 [&::-webkit-details-marker]:hidden">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary"><GraduationCap aria-hidden="true" className="size-6" /></span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2 font-semibold"><span>Known domains <span className="sr-only">({domainIds.length}/{new Set(benchmarks.map(b => b.id)).size})</span></span><span aria-hidden="true" className="rounded-full bg-primary/15 px-2 py-0.5 text-xs text-primary">{domainIds.length}/{new Set(benchmarks.map(b => b.id)).size}</span></span>
          <span className="mt-1 block text-xs text-muted-foreground">Disclosed source-family or related-domain training; S1MB evaluation-example use is not established by this label.</span>
        </span>
        <ChevronDown aria-hidden="true" className="size-4 shrink-0 text-primary group-open:rotate-180" />
      </summary>
      <p className="text-xs text-muted-foreground">{knownDomainContext} Actual evaluation-example use is labeled separately as ‘Evaluation examples used’. {disclosureContext}</p>
      <ul className="space-y-2">
        {domainIds.map(id => {
          const benchmark = benchmarks.find(b => b.id === id)!;
          const disclosures = entries.filter(e => e.benchmark_id === id);
          return <li key={id} ref={focus === id ? target : undefined} tabIndex={focus === id ? -1 : undefined} data-training-domain={id} className={`rounded-md border p-3 text-xs outline-none [overflow-wrap:anywhere] ${focus === id ? 'border-primary bg-primary/10 ring-2 ring-primary' : 'bg-muted/20'}`}>
            <details open={focus === id || undefined} className="space-y-2">
              <summary className="cursor-pointer font-medium">{benchmarkName(benchmark)} · {benchmark.task === 'noul' ? 'Noul' : benchmark.task === 'choice' ? 'Choice' : 'Score'}</summary>
              <p className="font-mono text-muted-foreground">{id}</p>
              {disclosures.map((e, i) => <div key={i} className="space-y-1">
                <p className="font-semibold">{exposureRelationship[e.relationship]}</p>
                <p>{e.note}</p>
                <p className="text-muted-foreground">{e.dataset ?? 'Training Dataset unknown / not disclosed'} · {e.config ?? 'Config unknown / not disclosed'} · {e.split ?? 'Split unknown / not disclosed'}</p>
                <p className="font-mono text-muted-foreground">Dataset revision: {e.revision ?? 'Unknown / not disclosed'}</p>
                <a className="underline underline-offset-4" href={e.evidence_url} target="_blank" rel="noreferrer">Training evidence ↗</a>
              </div>)}
            </details>
          </li>;
        })}
      </ul>
    </details>}
  </section>;
}
