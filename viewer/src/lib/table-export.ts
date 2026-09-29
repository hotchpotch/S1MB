export type ExportTable = { headers: string[]; rows: string[][] };

/** Capture rendered cells in their current order, excluding selection controls. */
export function readDisplayedTable(table: HTMLTableElement): ExportTable {
  const columns = Array.from(table.tHead?.rows[0]?.cells ?? []).flatMap((cell, index) => {
    if (index === 0 || getComputedStyle(cell).display === 'none') return [];
    return [{ index, labels: cell.dataset.exportColumns?.split(',') ?? [cell.innerText.replace(/\s+/g, ' ').trim()] }];
  });
  return {
    headers: columns.flatMap(column => column.labels),
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
  return [table.headers, ...table.rows].map(row => row.map(escape).join(',')).join('\r\n') + '\r\n';
}

export function tableMarkdown(table: ExportTable): string {
  const escape = (value: string) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('\\', '\\\\').replaceAll('|', '\\|').replaceAll('`', '\\`').replaceAll('*', '\\*').replaceAll('_', '\\_')
    .replaceAll('[', '\\[').replaceAll(']', '\\]').replace(/\r\n|\r|\n/g, '<br>');
  const line = (row: string[]) => `| ${row.map(escape).join(' | ')} |`;
  return [line(table.headers), `| ${table.headers.map(() => '---').join(' | ')} |`, ...table.rows.map(line)].join('\n') + '\n';
}
