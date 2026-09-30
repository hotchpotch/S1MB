/** One immutable, bundled display artifact per server lifetime. No source scans or remote I/O. */
import { readDisplay } from './display-data';
import type { Snapshot } from './types';
const shared = globalThis as typeof globalThis & { s1mbDisplay?: Promise<Snapshot> };
export function getSnapshot(): Promise<Snapshot> {
  const file = process.env.S1MB_DISPLAY_FILE;
  if (!file) throw new Error('Start with npm start -- --display-file /path/to/viewer-summary.json');
  return shared.s1mbDisplay ??= readDisplay(file).then(({ snapshot }) => snapshot).catch(error => {
    shared.s1mbDisplay = undefined;
    throw error;
  });
}
