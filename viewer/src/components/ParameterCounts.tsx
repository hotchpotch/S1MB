"use client";

import { Tooltip } from 'radix-ui';
import type { ModelInfo } from '../lib/types';

function formatCount(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value) || value < 0) return '—';
  const millions = value / 1_000_000;
  return `${(millions >= 1000 ? millions / 1000 : millions).toLocaleString('en-US', { maximumFractionDigits: millions >= 1000 ? 2 : 0 })}${millions >= 1000 ? 'B' : 'M'}`;
}

export function ParameterCountsHeader() {
  return <Tooltip.Provider delayDuration={150}><Tooltip.Root>
    <Tooltip.Trigger asChild><button type="button" aria-label="TP: Total Parameters; AP: Active Parameters" className="cursor-help border-b border-dotted border-muted-foreground text-right leading-tight"><span className="block">TP</span><span className="block">AP</span></button></Tooltip.Trigger>
    <Tooltip.Portal><Tooltip.Content sideOffset={6} className="z-50 max-w-80 rounded-md border bg-popover p-3 text-xs text-popover-foreground shadow-md">
      <p><strong>TP — Total Parameters.</strong> The model’s total parameter count.</p>
      <p className="mt-2"><strong>AP — Active Parameters.</strong> The reported count excludes lookup-only embeddings and retains shared output weights.</p>
      <p className="mt-2">M = million; B = billion (1,000M = 1B). M values are rounded to whole millions; B values to two decimal places. — means unknown.</p>
      <Tooltip.Arrow className="fill-popover" />
    </Tooltip.Content></Tooltip.Portal>
  </Tooltip.Root></Tooltip.Provider>;
}

export function ParameterCounts({ model }: { model: ModelInfo }) {
  return <div className="font-mono tabular-nums text-right leading-tight whitespace-nowrap">
    <span className="block" title={`Total Parameters: ${model.total_params?.toLocaleString('en-US') ?? 'Unknown'}`}><span className="sr-only">TP: </span>{formatCount(model.total_params)}</span>
    <span className="block text-muted-foreground" title={`Active Parameters: ${model.active_params?.toLocaleString('en-US') ?? 'Unknown'}`}><span className="sr-only">AP: </span>{formatCount(model.active_params)}</span>
  </div>;
}
