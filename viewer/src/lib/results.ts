/** The web process holds compact summaries; a child process reads and parses files. */
import { spawn, type ChildProcess } from 'node:child_process';
import path from 'node:path';
import type { Snapshot } from './types';

export type Refresh = () => Promise<Snapshot | null>;
export class SnapshotCache {
  private current?: Snapshot;
  private pending?: Promise<Snapshot>;
  private nextCheck = 0;
  constructor(private refresh: Refresh, private intervalMs: number, private now = Date.now) {}
  get(): Promise<Snapshot> {
    if (!this.current) return this.pending ?? this.start();
    // Do not await refresh: the same request and concurrent requests get the old snapshot.
    if (!this.pending && this.now() >= this.nextCheck) void this.start().catch(() => {});
    return Promise.resolve(this.current);
  }
  private start(): Promise<Snapshot> {
    this.nextCheck = this.now() + this.intervalMs;
    this.pending = this.refresh().then(snapshot => {
      if (!snapshot && !this.current) throw new Error('Worker returned no initial snapshot');
      this.current = { ...(snapshot ?? this.current!), cache: { checkedAt: new Date(this.now()).toISOString(), refreshFailed: false } };
      return this.current;
    }).catch(error => {
      console.error('Results refresh failed:', error instanceof Error ? error.message : 'unknown error');
      if (!this.current) throw error;
      this.current = { ...this.current, cache: { ...this.current.cache!, refreshFailed: true } };
      return this.current;
    }).finally(() => { this.pending = undefined; this.nextCheck = this.now() + this.intervalMs; });
    return this.pending;
  }
}

export class ResultsWorker {
  private child?: ChildProcess;
  constructor(private dataDir: string, private dirs: string[]) {}
  refresh(): Promise<Snapshot | null> {
    if (this.child && !this.child.connected) this.child = undefined;
    const child = this.child ??= spawn(process.execPath, ['--import', 'tsx', path.join(process.cwd(), 'scripts/results-worker.ts')], {
      stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
    });
    return new Promise((resolve, reject) => {
      const cleanup = () => { clearTimeout(timer); child.off('message', message); child.off('exit', exit); child.off('error', error); };
      const error = () => { cleanup(); this.child = undefined; child.kill(); reject(new Error('Results worker failed')); };
      const exit = () => { cleanup(); this.child = undefined; reject(new Error('Results worker exited')); };
      const message = (value: { snapshot?: Snapshot | null; error?: string }) => {
        cleanup();
        if (value.error) reject(new Error(value.error)); else resolve(value.snapshot ?? null);
      };
      const timer = setTimeout(error, 15 * 60_000);
      child.once('message', message); child.once('exit', exit); child.once('error', error);
      child.send({ dataDir: this.dataDir, dirs: this.dirs }, e => { if (e) error(); });
    });
  }
  close() { this.child?.kill(); this.child = undefined; }
}

const shared = globalThis as typeof globalThis & { s1mbFiles?: SnapshotCache };
export function getSnapshot(): Promise<Snapshot> {
  if (!shared.s1mbFiles) {
    const dataDir = process.env.S1MB_DATA_DIR;
    if (!dataDir) throw new Error('Start with npm start so the runtime directories are configured');
    const dirs: unknown = JSON.parse(process.env.S1MB_RESULTS_DIRS ?? '[]');
    if (!Array.isArray(dirs) || !dirs.length || dirs.some(d => typeof d !== 'string')) throw new Error('Configure at least one results directory');
    const seconds = Number(process.env.S1MB_RESULTS_CHECK_SECONDS ?? 0);
    if (!Number.isSafeInteger(seconds) || seconds < 0 || seconds > 86400) throw new Error('Invalid filesystem check interval');
    const worker = new ResultsWorker(dataDir, dirs);
    shared.s1mbFiles = new SnapshotCache(() => worker.refresh(), seconds * 1000);
  }
  return shared.s1mbFiles.get();
}
