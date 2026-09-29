import type { ReactNode } from 'react';
import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react';
import { cn } from '../lib/utils';

export function SortIcon({ direction }: { direction: 'ascending' | 'descending' | 'none' }) {
  const Icon = direction === 'none' ? ArrowUpDown : direction === 'ascending' ? ArrowUp : ArrowDown;
  return <Icon aria-hidden="true" className={cn('ml-0.5 size-3 shrink-0', direction === 'none' ? 'opacity-40' : 'text-primary')} />;
}

/** Metric preference is independent of the current table sort direction. */
export function MetricDirection({ direction, children }: { direction: 'up' | 'down'; children: ReactNode }) {
  const Icon = direction === 'up' ? ArrowUp : ArrowDown;
  return <span className="inline-flex items-center gap-0.5">
    <span>{children}</span><Icon aria-hidden="true" className="size-3 shrink-0" />
    <span className="sr-only">{direction === 'up' ? ' (higher is better)' : ' (lower is better)'}</span>
  </span>;
}
