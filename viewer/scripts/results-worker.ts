/** Keep cache I/O, filesystem scans and XZ decoding off the request process. */
import { FilesystemResults } from '../src/lib/result-loader';
import { SnapshotStore } from '../src/lib/snapshot-store';
import type { Snapshot } from '../src/lib/types';
let loader: FilesystemResults | undefined;
let store: SnapshotStore | undefined;
let pendingSave: Snapshot | undefined;
let busy = false;
process.on('message', async ({ action, dataDir, dirs, cacheDir }: { action: 'restore' | 'refresh'; dataDir: string; dirs: string[]; cacheDir?: string }) => {
  if (busy) return;
  busy = true;
  try {
    if (cacheDir) store ??= new SnapshotStore(cacheDir, dataDir, dirs);
    if (action === 'restore') {
      process.send?.({ snapshot: await store?.restore() ?? null });
      return;
    }
    loader ??= new FilesystemResults(dataDir, dirs);
    const snapshot = await loader.refresh();
    if (snapshot) pendingSave = snapshot;
    if (store && pendingSave) {
      try { await store.save(pendingSave); pendingSave = undefined; }
      catch { console.error('Results cache persistence failed; will retry on the next check.'); }
    }
    process.send?.({ snapshot });
  } catch (error) {
    process.send?.({ error: error instanceof Error ? error.message : 'Results refresh failed' });
  } finally { busy = false; }
});
process.on('disconnect', () => process.exit(0));
