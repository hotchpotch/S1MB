"use client";

import type { RefObject } from 'react';
import { Download, FileText } from 'lucide-react';
import { readDisplayedTable, tableCsv, tableMarkdown } from '../lib/table-export';
import { Button } from './ui/button';

export function TableExports({ tableRef, filename, disabled }: {
  tableRef: RefObject<HTMLTableElement | null>; filename: string; disabled: boolean;
}) {
  function exportTable(markdown: boolean) {
    if (!tableRef.current) return;
    const table = readDisplayedTable(tableRef.current);
    const blob = new Blob([markdown ? tableMarkdown(table) : '\uFEFF' + tableCsv(table)], {
      type: markdown ? 'text/plain;charset=utf-8' : 'text/csv;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    if (markdown) { anchor.target = '_blank'; anchor.rel = 'noopener noreferrer'; }
    else anchor.download = `${filename}.csv`;
    document.body.append(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  }
  return <>
    <Button variant="ghost" size="xs" aria-label="Download CSV" title="Download CSV" disabled={disabled} onClick={() => exportTable(false)}><Download className="size-3" />CSV</Button>
    <Button variant="ghost" size="xs" aria-label="View Markdown" title="View Markdown as plain text" disabled={disabled} onClick={() => exportTable(true)}><FileText className="size-3" />Markdown</Button>
  </>;
}
