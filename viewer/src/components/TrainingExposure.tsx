"use client";

import { ChevronDown, GraduationCap, TriangleAlert, CircleHelp } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { Benchmark, ModelInfo } from '../lib/types';
import { exposuresFor, exposureLabel, exposureRelationship, registrationSourceLabel, datasetAccessLabel } from '../lib/training-exposure';
import { benchmarkName } from '../lib/comparison';
import { ColumnHelp } from './ColumnHelp';

const knownDomainContext = 'These entries describe reported source-family or related-domain training. Inspect each entry’s evidence and limitations; unlisted benchmarks remain Unknown / not disclosed.';

const disclosureContext = 'Disclosures do not change scores or rankings.';
const unknownContext = 'Training exposure is unknown / not disclosed for this scope. The model may still have trained on these domains. Unknown does not mean unused or independently audited.';
const registrationContext = 'Contributor reviewed sources means a contributor added or amended the entry from public materials or an author statement. It is not an independent audit. Public data availability and actual training use are separate facts.';

/** Reported training exposure does not modify scores or prove independence. */
export function TrainingBadge({ model, benchmarkId, benchmarkIds, onClick }: { model: ModelInfo; benchmarkId?: string; benchmarkIds?: string[]; onClick?: () => void }) {
  const entries = exposuresFor(model, benchmarkId).filter(e => !benchmarkIds || benchmarkIds.includes(e.benchmark_id));
  const known = new Set(entries.map(e => e.benchmark_id)).size;
  const total = benchmarkId ? 1 : benchmarkIds ? new Set(benchmarkIds).size : null;
  const coverage = total == null ? `${known} benchmarks` : `${known}/${total} benchmarks in this scope`;
  const label = exposureLabel(entries);
  const Icon = label === 'Unknown' ? CircleHelp : label === 'Known domain' ? GraduationCap : TriangleAlert;
  return <span className={`inline-flex align-middle ${entries.length ? 'text-primary' : 'text-muted-foreground'}`} data-training-exposure={benchmarkId ?? 'model'}>
    <ColumnHelp label={`${label}${entries.length ? `: ${coverage}` : ''} · ${benchmarkId ?? model.display_name ?? model.id}`} onClick={onClick}
      description={<><p className="font-semibold">{label}{entries.length ? ` · ${coverage}` : ''}</p><p>{entries.length ? 'Inspect the declaration and linked sources; this is not a trust rating.' : unknownContext}</p><p>{onClick ? 'Click to view training disclosures.' : 'Training disclosure information.'}</p></>}>
      <span className="inline-flex items-center gap-1 whitespace-nowrap"><Icon aria-hidden="true" className="size-3.5 shrink-0" />
        {!entries.length && <span className="text-xs font-normal">Unknown</span>}
      </span>
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
    {!entries.length ? <div className="space-y-2 rounded-lg border p-4 text-muted-foreground"><p className="flex items-center gap-2 font-semibold"><CircleHelp aria-hidden="true" className="size-5" /> Unknown</p><p className="text-sm">{unknownContext}</p></div> : <details ref={panel} tabIndex={-1} open={expanded} onToggle={event => setExpanded(event.currentTarget.open)} data-training-focus={selected ? focus : undefined} className={`group rounded-lg border border-primary/30 outline-none bg-primary/5 p-4 open:space-y-4 ${selected ? 'ring-2 ring-primary ring-offset-2 ring-offset-card' : ''}`}>
      <summary className="flex cursor-pointer list-none items-center gap-3 rounded-sm focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-4 [&::-webkit-details-marker]:hidden">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary"><GraduationCap aria-hidden="true" className="size-6" /></span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2 font-semibold"><span>Known domains <span className="sr-only">({domainIds.length}/{new Set(benchmarks.map(b => b.id)).size})</span></span><span aria-hidden="true" className="rounded-full bg-primary/15 px-2 py-0.5 text-xs text-primary">{domainIds.length}/{new Set(benchmarks.map(b => b.id)).size}</span></span>
          <span className="mt-1 block text-xs text-muted-foreground">Disclosed source-family or related-domain training; S1MB evaluation-example use is not established by this label.</span>
        </span>
        <ChevronDown aria-hidden="true" className="size-4 shrink-0 text-primary group-open:rotate-180" />
      </summary>
      <p className="text-xs text-muted-foreground">{knownDomainContext} Actual evaluation-example use is labeled separately as ‘Evaluation examples used’. {disclosureContext}</p>
      <p className="text-xs text-muted-foreground">{registrationContext}</p>
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
                <p>Registration source: {registrationSourceLabel[e.registration_source ?? 'unknown']}</p>
                <p>Training dataset access: {datasetAccessLabel[e.dataset_access ?? 'unknown']}</p>
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
