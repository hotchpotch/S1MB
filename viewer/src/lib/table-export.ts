export type ExportTable = { headers: string[]; rows: string[][]; links?: { url: string; huggingface: string }[] };

/** Capture rendered cells in their current order, excluding selection controls. */
export function readDisplayedTable(table: HTMLTableElement): ExportTable {
  const columns = Array.from(table.tHead?.rows[0]?.cells ?? []).flatMap((cell, index) => {
    if (index === 0 || getComputedStyle(cell).display === 'none') return [];
    return [{ index, labels: cell.dataset.exportColumns?.split(',') ?? [cell.innerText.replace(/\s+/g, ' ').trim()] }];
  });
  return {
    headers: columns.flatMap(column => column.labels),
    links: Array.from(table.tBodies[0]?.rows ?? []).map(row => {
      const cell = row.querySelector<HTMLElement>('[data-export-url], [data-export-huggingface]');
      return { url: cell?.dataset.exportUrl ?? '', huggingface: cell?.dataset.exportHuggingface ?? '' };
    }),
    rows: Array.from(table.tBodies[0]?.rows ?? []).map(row => columns.flatMap(column => {
      const cell = row.cells[column.index];
      const values = Array.from(cell.querySelectorAll<HTMLElement>('[data-export-value]'));
      return values.length ? values.map(value => value.dataset.exportValue!) : [cell.innerText.trim()];
    })),
  };
}

export function tableCsv(table: ExportTable): string {
  const escape = (value: string) => {
    // Prevent model-authored text from becoming spreadsheet formulas.
    const safe = /^[\s]*[=+@-]/.test(value) ? `'${value}` : value;
    return `"${safe.replaceAll('"', '""')}"`;
  };
  const rows = table.rows.map((row, index) => [...row, table.links?.[index]?.url ?? '', table.links?.[index]?.huggingface ?? '']);
  return [[...table.headers, 'url', 'huggingface'], ...rows].map(row => row.map(escape).join(',')).join('\r\n') + '\r\n';
}

export function tableMarkdown(table: ExportTable): string {
  const escape = (value: string) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('\\', '\\\\').replaceAll('|', '\\|').replaceAll('`', '\\`').replaceAll('*', '\\*').replaceAll('_', '\\_')
    .replaceAll('[', '\\[').replaceAll(']', '\\]').replace(/\r\n|\r|\n/g, '<br>');
  const line = (row: string[]) => `| ${row.join(' | ')} |`;
  const rows = table.rows.map((row, index) => row.map((value, column) => {
    const label = escape(value);
    const links = table.links?.[index];
    const url = links?.huggingface || links?.url;
    if (table.headers[column] !== 'Model' || !url || !/^https?:\/\//i.test(url)) return label;
    const destination = url.replace(/[\s<>|()\\]/g, character => encodeURIComponent(character).replace('(', '%28').replace(')', '%29'));
    return `[${label}](${destination})`;
  }));
  return [line(table.headers.map(escape)), `| ${table.headers.map(() => '---').join(' | ')} |`, ...rows.map(line)].join('\n') + '\n';
}
