"use client";

import { ColumnHelp } from './ColumnHelp';
import type { ModelInfo } from '../lib/types';

function formatCount(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value) || value < 0) return '—';
  const millions = value / 1_000_000;
  return `${(millions >= 1000 ? millions / 1000 : millions).toLocaleString('en-US', { maximumFractionDigits: millions >= 1000 ? 2 : 0 })}${millions >= 1000 ? 'B' : 'M'}`;
}

export function ParameterCountsHeader() {
  return <ColumnHelp label="TP: Total Parameters; AP: Active Parameters" description={<>
    <p><strong>TP — Total Parameters.</strong> The model’s total parameter count.</p>
    <p className="mt-2"><strong>AP — Active Parameters.</strong> The reported count excludes lookup-only embeddings and retains shared output weights.</p>
    <p className="mt-2">AP here counts non-lookup parameters, not per-input active parameters in a routed MoE model or memory usage. Large multilingual vocabulary embeddings can make AP much smaller than TP; those embeddings are still used during inference.</p>
    <p className="mt-2">M = million; B = billion (1,000M = 1B). M values are rounded to whole millions; B values to two decimal places. — means unknown.</p>
  </>}><span className="block">TP</span><span className="block">AP</span></ColumnHelp>;
}

export function ParameterCounts({ model }: { model: ModelInfo }) {
  return <div className="font-mono tabular-nums text-[10px] text-right leading-tight whitespace-nowrap">
    <span className="block" title={`Total Parameters: ${model.total_params?.toLocaleString('en-US') ?? 'Unknown'}`}><span className="sr-only">TP: </span>{formatCount(model.total_params)}</span>
    <span className="block text-muted-foreground" title={`Active Parameters: ${model.active_params?.toLocaleString('en-US') ?? 'Unknown'}`}><span className="sr-only">AP: </span>{formatCount(model.active_params)}</span>
  </div>;
}
