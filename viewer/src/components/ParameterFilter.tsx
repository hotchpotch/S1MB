"use client";

import { ColumnHelp } from './ColumnHelp';
import { cn } from '../lib/utils';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { ALL_PARAMETERS, parameterLabel, rangeActive, type ParameterRange } from '../lib/parameter-filters';

export function ParameterFilter({ label, description, value, onChange }: {
  label: string; description: string; value: ParameterRange; onChange: (value: ParameterRange) => void;
}) {
  const active = rangeActive(value);
  return <div className="min-w-0">
    <div className="flex min-h-6 flex-wrap items-center gap-x-2 gap-y-1 text-xs">
      <ColumnHelp label={label} description={description}><span className="font-medium">{label}</span></ColumnHelp>
      <span className={cn("ml-auto font-mono text-[11px] tabular-nums", active ? "text-primary" : "text-muted-foreground")}>{active ? `${parameterLabel(value[0])} – ${parameterLabel(value[1])}` : 'All sizes'}</span>
      {active && <Button variant="ghost" size="sm" className="h-6 px-1.5 text-[11px] text-muted-foreground" onClick={() => onChange([...ALL_PARAMETERS])}>Reset</Button>}
    </div>
    <Slider min={0} max={100} step={0.1} minStepsBetweenThumbs={1} value={value} onValueChange={next => onChange(next as ParameterRange)} thumbLabels={[`${label} minimum`, `${label} maximum`]} thumbValueTexts={[value[0] === 0 ? "No minimum" : parameterLabel(value[0]), value[1] === 100 ? "No maximum" : parameterLabel(value[1])]} />
    <div aria-hidden="true" className="flex justify-between font-mono text-[10px] text-muted-foreground"><span>&lt;50M</span><span>1B</span><span>5B</span><span>10B</span><span>20B</span><span>35B+</span></div>
  </div>;
}
