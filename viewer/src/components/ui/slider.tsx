"use client";

import { Slider as SliderPrimitive } from 'radix-ui';
import type { ComponentProps } from 'react';
import { cn } from '@/lib/utils';

export function Slider({ className, thumbLabels, thumbValueTexts, ...props }: ComponentProps<typeof SliderPrimitive.Root> & { thumbLabels: [string, string]; thumbValueTexts: [string, string] }) {
  return <SliderPrimitive.Root {...props} className={cn('relative flex h-7 w-full touch-none select-none items-center', className)}>
    <SliderPrimitive.Track className="relative h-1 grow overflow-hidden rounded-full bg-border">
      <SliderPrimitive.Range className="absolute h-full bg-primary/60" />
    </SliderPrimitive.Track>
    {thumbLabels.map((label, index) => <SliderPrimitive.Thumb key={label} aria-label={label} aria-valuetext={thumbValueTexts[index]} className="relative block size-3.5 rounded-full border border-primary/70 bg-background shadow-xs outline-none transition-shadow after:absolute after:-inset-2 after:rounded-full hover:ring-4 hover:ring-primary/10 focus-visible:ring-[3px] focus-visible:ring-ring/50" />)}
  </SliderPrimitive.Root>;
}
