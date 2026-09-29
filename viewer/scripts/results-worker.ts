/** Keep filesystem I/O, JSON parsing and XZ decoding off the request process. */
import { FilesystemResults } from '../src/lib/result-loader';
let loader: FilesystemResults | undefined;
let busy = false;
process.on('message', async ({ dataDir, dirs }: { dataDir: string; dirs: string[] }) => {
  if (busy) return;
  busy = true;
  try {
    loader ??= new FilesystemResults(dataDir, dirs);
    process.send?.({ snapshot: await loader.refresh() });
  } catch (error) {
    process.send?.({ error: error instanceof Error ? error.message : 'Results refresh failed' });
  } finally { busy = false; }
});
process.on('disconnect', () => process.exit(0));
