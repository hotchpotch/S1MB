"use client";

import type { ReactNode } from 'react';
import { Tooltip } from 'radix-ui';

/** Shared, keyboard-accessible help for result table headings. */
export function ColumnHelp({ children, description, label, onClick }: {
  children: ReactNode; description: ReactNode; label?: string; onClick?: () => void;
}) {
  return <Tooltip.Provider delayDuration={150}><Tooltip.Root>
    <Tooltip.Trigger asChild>
      <button type="button" aria-label={label} onClick={onClick} className={`${onClick ? 'cursor-pointer' : 'cursor-help'} text-inherit font-bold leading-tight [text-align:inherit] no-underline focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-2`}>{children}</button>
    </Tooltip.Trigger>
    <Tooltip.Portal><Tooltip.Content sideOffset={6} collisionPadding={12} className="z-50 max-w-[min(20rem,calc(100vw-24px))] rounded-md border bg-popover p-3 text-xs font-normal text-popover-foreground shadow-md">
      {description}<Tooltip.Arrow className="fill-popover" />
    </Tooltip.Content></Tooltip.Portal>
  </Tooltip.Root></Tooltip.Provider>;
}
