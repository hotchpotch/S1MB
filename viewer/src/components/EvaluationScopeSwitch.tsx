"use client";

import type { Category } from '../lib/types';
import { cn } from '../lib/utils';

/** Selects the evaluation dataset, rather than the interface language. */
export function EvaluationScopeSwitch({ categories, selected, onSelect }: {
  categories: Category[]; selected: string; onSelect: (id: string) => void;
}) {
  const scopes = categories.filter(c => ['english-v1', 'japanese-v1'].includes(c.id));
  const name = (id: string) => id === 'english-v1' ? 'English v1' : 'Japanese v1';
  return <nav aria-label="Evaluation language" className="flex flex-wrap items-center gap-2">
    <div className="inline-flex rounded-md border border-primary/30 bg-primary/5 p-0.5">
      {scopes.map(c => <button key={c.id} type="button" aria-pressed={c.id === selected} onClick={() => onSelect(c.id)} className={cn('rounded px-2 py-1.5 text-xs focus-visible:outline-2 focus-visible:outline-ring', c.id === selected ? 'bg-primary text-primary-foreground' : 'text-muted-foreground')}>{name(c.id)}</button>)}
    </div>
  </nav>;
}
