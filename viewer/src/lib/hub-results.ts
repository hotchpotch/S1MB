/** Revision-pinned Hub downloads and transactional, request-driven refreshes. */
import { createReadStream } from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';
import { copyFile, mkdir, mkdtemp, open, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import lockfile from 'proper-lockfile';
import { z } from 'zod';
import { metadataSchema, readResultJson } from './result-files';
import type { Snapshot } from './types';

const origin = 'https://huggingface.co';
const sha = z.string().regex(/^[a-f0-9]{40}$/);
const repo = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/);
const filePath = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]*__[A-Za-z0-9][A-Za-z0-9._-]*\/(?:metadata\.json|[A-Za-z0-9][A-Za-z0-9._-]*\.json\.xz)$/);
const maxBytes = 64 * 1024 * 1024;
const fileSchema = z.object({
  path: filePath, size: z.number().int().min(0).max(maxBytes), oid: sha,
  lfs: z.object({ oid: z.string().regex(/^[a-f0-9]{64}$/), size: z.number().int().min(0).max(maxBytes) }).optional(),
});
type HubFile = z.infer<typeof fileSchema>;
const stateSchema = z.object({
  revision: sha, generation: z.string().regex(/^snapshot-[A-Za-z0-9]+$/),
  checkedAt: z.number().finite().nonnegative(), files: z.array(fileSchema).max(10_000),
});
type State = z.infer<typeof stateSchema>;
export type HubResultsOptions = { repoId: string; revision: string; cacheDir: string; ttlMs: number; token?: string };

export function hubResultsOptions(env: Record<string, string | undefined> = process.env): HubResultsOptions | undefined {
  if (!env.S1MB_HF_RESULTS_REPO) return undefined;
  const ttl = Number(env.S1MB_HF_RESULTS_CACHE_SECONDS ?? 3600);
  if (!Number.isSafeInteger(ttl) || ttl < 3600 || ttl > 2_147_483) throw new Error('Results cache interval must be an integer between 3600 and 2147483 seconds');
  const revision = env.S1MB_HF_RESULTS_REVISION || 'main';
  if (revision.length > 200 || /[\x00-\x20\x7f]/.test(revision)) throw new Error('Invalid results revision');
  return {
    repoId: repo.parse(env.S1MB_HF_RESULTS_REPO), revision,
    cacheDir: path.resolve(/* turbopackIgnore: true */ env.S1MB_HF_RESULTS_CACHE_DIR || path.join(tmpdir(), 's1mb-viewer-results')),
    ttlMs: ttl * 1000, token: env.HF_TOKEN || undefined,
  };
}

export class HubRequestError extends Error {
  constructor(readonly status: number, readonly retryAfterMs = 0) { super(`Hugging Face request failed (HTTP ${status})`); }
}

/** Fetch bodies with hard limits; never expose response bodies, tokens or signed URLs in errors. */
export class HubClient {
  constructor(private options: HubResultsOptions, private fetcher: typeof fetch = fetch) {}
  private async request(url: string): Promise<Response> {
    let response: Response;
    try {
      response = await this.fetcher(url, {
        headers: this.options.token ? { Authorization: `Bearer ${this.options.token}` } : {},
        signal: AbortSignal.timeout(60_000), cache: 'no-store',
      });
    } catch { throw new Error('Hugging Face request timed out or could not connect'); }
    if (!response.ok) {
      const retry = response.headers.get('retry-after');
      const seconds = retry === null ? NaN : Number(retry);
      const reset = response.headers.get('ratelimit')?.match(/\bt=(\d+)/)?.[1];
      const delay = Math.max(0, Number.isFinite(seconds) ? seconds * 1000 : Date.parse(retry ?? '') - Date.now() || 0,
        Number(reset ?? 0) * 1000);
      await response.body?.cancel();
      throw new HubRequestError(response.status, delay);
    }
    return response;
  }
  private async json(url: string): Promise<{ value: unknown; next: string | undefined }> {
    const response = await this.request(url);
    const reader = response.body?.getReader();
    if (!reader) throw new Error('Empty Hub response');
    const chunks: Uint8Array[] = [];
    let bytes = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        bytes += value.length;
        if (bytes > 8 * 1024 * 1024) throw new Error('Hub metadata exceeds size limit');
        chunks.push(value);
      }
      return { value: JSON.parse(Buffer.concat(chunks).toString('utf8')),
        next: response.headers.get('link')?.match(/<([^>]+)>;\s*rel="next"/)?.[1] };
    } finally { await reader.cancel(); }
  }
  async revision(): Promise<string> {
    const { value } = await this.json(`${origin}/api/datasets/${this.options.repoId}/revision/${encodeURIComponent(this.options.revision)}?expand=sha`);
    return z.object({ sha }).parse(value).sha;
  }
  async files(revision: string): Promise<HubFile[]> {
    const base = `${origin}/api/datasets/${this.options.repoId}/tree/${revision}`;
    let url: string | undefined = `${base}?recursive=true&expand=false`;
    const visited = new Set<string>();
    const files: HubFile[] = [];
    while (url) {
      const parsed = new URL(url);
      if (parsed.origin !== origin || parsed.pathname !== new URL(base).pathname || visited.has(url) || visited.size >= 100) throw new Error('Invalid Hub pagination');
      visited.add(url);
      const page = await this.json(url);
      const entries = z.array(z.object({ type: z.string(), path: z.string() }).passthrough()).parse(page.value);
      for (const entry of entries) {
        if (entry.type !== 'file') continue;
        // Repository cards and attributes are not measurements. Reject malformed submission paths.
        if (!entry.path.endsWith('.json') && !entry.path.endsWith('.json.xz')) continue;
        files.push(fileSchema.parse(entry));
      }
      if (files.length > 10_000) throw new Error('Too many Hub result files');
      url = page.next;
    }
    if (new Set(files.map(f => f.path)).size !== files.length) throw new Error('Duplicate Hub file paths');
    if (files.reduce((n, f) => n + f.size, 0) > 2 * 1024 ** 3) throw new Error('Hub result snapshot exceeds 2 GiB');
    const paths = new Set(files.map(f => f.path));
    for (const f of files) if (!paths.has(`${f.path.split('/')[0]}/metadata.json`)) throw new Error('Published result requires metadata.json');
    if (!files.some(f => f.path.endsWith('.json.xz'))) throw new Error('No published results found');
    return files;
  }
  async download(revision: string, file: HubFile, destination: string): Promise<void> {
    const response = await this.request(`${origin}/datasets/${this.options.repoId}/resolve/${revision}/${file.path}`);
    const reader = response.body?.getReader();
    if (!reader) throw new Error('Empty Hub download');
    const hash = createHash(file.lfs ? 'sha256' : 'sha1');
    if (!file.lfs) hash.update(`blob ${file.size}\0`);
    let bytes = 0;
    const handle = await open(destination, 'wx', 0o600);
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        bytes += value.length;
        if (bytes > file.size || bytes > maxBytes) throw new Error('Hub download exceeds declared size');
        hash.update(value);
        await handle.writeFile(value);
      }
      if (bytes !== file.size || (file.lfs && file.lfs.size !== bytes) || hash.digest('hex') !== (file.lfs?.oid ?? file.oid)) throw new Error('Hub download checksum or size mismatch');
    } finally { await handle.close(); await reader.cancel(); }
  }
}

export class HubResultsCache {
  private current?: Snapshot;
  private generation?: string;
  private nextCheck = 0;
  private pending?: Promise<Snapshot>;
  private root: string;
  constructor(
    private options: HubResultsOptions,
    private load: (directory: string) => Promise<Snapshot>,
    private client = new HubClient(options),
    private now = Date.now,
  ) {
    this.root = path.join(options.cacheDir, createHash('sha256').update(`${options.repoId}@${options.revision}`).digest('hex'));
  }
  get(): Promise<Snapshot> {
    if (this.pending) return this.pending;
    if (this.current && this.now() < this.nextCheck) return Promise.resolve(this.current);
    this.pending = this.refresh().finally(() => { this.pending = undefined; });
    return this.pending;
  }
  private async readState(): Promise<State | undefined> {
    try { return stateSchema.parse(JSON.parse(await readFile(path.join(this.root, 'state.json'), 'utf8'))); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined; throw error; }
  }
  private async saveState(state: State): Promise<void> {
    const temporary = path.join(this.root, `.state-${randomUUID()}`);
    try {
      await writeFile(temporary, JSON.stringify(state), { mode: 0o600, flag: 'wx' });
      await rename(temporary, path.join(this.root, 'state.json'));
    } finally { await rm(temporary, { force: true }); }
  }
  private async prune(keep: string[]): Promise<void> {
    for (const entry of await readdir(this.root, { withFileTypes: true })) {
      if (entry.isDirectory() && /^snapshot-[A-Za-z0-9]+$/.test(entry.name) && !keep.includes(entry.name)) {
        await rm(path.join(this.root, entry.name), { recursive: true, force: true });
      }
    }
  }
  private async validate(state: State): Promise<Snapshot> {
    const directory = path.join(this.root, state.generation);
    // Recheck cached bytes too: the displayed revision must describe the actual files.
    for (const file of state.files) {
      const filename = path.join(directory, file.path);
      if ((await stat(filename)).size !== file.size) throw new Error('Cached result size mismatch');
      const hash = createHash(file.lfs ? 'sha256' : 'sha1');
      if (!file.lfs) hash.update(`blob ${file.size}\0`);
      for await (const chunk of createReadStream(filename)) hash.update(chunk);
      if (hash.digest('hex') !== (file.lfs?.oid ?? file.oid)) throw new Error('Cached result checksum mismatch');
    }
    for (const file of state.files.filter(f => f.path.endsWith('/metadata.json'))) {
      const metadata = metadataSchema.parse(await readResultJson(path.join(directory, file.path)));
      if (metadata.model_id !== file.path.split('/')[0]) throw new Error('Model directory/metadata ID mismatch');
    }
    const snapshot = await this.load(directory);
    if (snapshot.issues.length || snapshot.results.length !== state.files.filter(f => f.path.endsWith('.json.xz')).length) {
      throw new Error(`Downloaded results failed validation. Ensure all recorded evaluation dataset revisions are installed. ${snapshot.issues.join('; ')}`);
    }
    const url = `${origin}/datasets/${this.options.repoId}`;
    snapshot.resultsSource = { repoId: this.options.repoId, revision: state.revision, url: `${url}/tree/${state.revision}`, checkedAt: new Date(state.checkedAt).toISOString() };
    for (const result of snapshot.results) result.resultUrl = `${url}/blob/${state.revision}/${result.run_id}/${result.benchmark.id}.json.xz`;
    return snapshot;
  }
  private async refresh(): Promise<Snapshot> {
    let release: (() => Promise<void>) | undefined;
    try {
      await mkdir(this.root, { recursive: true, mode: 0o700 });
      release = await lockfile.lock(this.root, { retries: { retries: 120, minTimeout: 500, maxTimeout: 1000 }, stale: 120_000 });
      const state = await this.readState();
      if (state && this.generation !== state.generation) {
        this.current = await this.validate(state);
        this.generation = state.generation;
      }
      if (state && this.current && state.checkedAt <= this.now() && this.now() - state.checkedAt < this.options.ttlMs) {
        this.current = { ...this.current, resultsSource: { ...this.current.resultsSource!, checkedAt: new Date(state.checkedAt).toISOString(), refreshFailed: false } };
        this.nextCheck = state.checkedAt + this.options.ttlMs;
        return this.current;
      }
      this.nextCheck = this.now() + this.options.ttlMs;
      const revision = await this.client.revision();
      if (state?.revision === revision && this.current) {
        const updated = { ...state, checkedAt: this.now() };
        await this.saveState(updated);
        this.nextCheck = updated.checkedAt + this.options.ttlMs;
        this.current = { ...this.current, resultsSource: { ...this.current.resultsSource!, checkedAt: new Date(updated.checkedAt).toISOString(), refreshFailed: false } };
        return this.current;
      }
      const files = await this.client.files(revision);
      const directory = await mkdtemp(path.join(this.root, 'snapshot-'));
      let installed = false;
      try {
        const previous = new Map(state?.files.map(f => [f.path, f]));
        // Four bounded downloads at a time. Wait for the whole batch before cleaning up on failure.
        for (let i = 0; i < files.length; i += 4) {
          const batch = await Promise.allSettled(files.slice(i, i + 4).map(async file => {
            const destination = path.join(directory, file.path);
            await mkdir(path.dirname(destination), { recursive: true });
            const old = previous.get(file.path);
            if (state && old && old.oid === file.oid && old.lfs?.oid === file.lfs?.oid && old.size === file.size) {
              const source = path.join(this.root, state.generation, file.path);
              if (await stat(source).then(s => s.size === file.size, () => false)) {
                await copyFile(source, destination);
                return;
              }
            }
            await this.client.download(revision, file, destination);
          }));
          const failure = batch.find(r => r.status === 'rejected');
          if (failure?.status === 'rejected') throw failure.reason;
        }
        const updated: State = { revision, generation: path.basename(directory), checkedAt: this.now(), files };
        const candidate = await this.validate(updated);
        await this.saveState(updated);
        installed = true;
        this.current = candidate;
        this.generation = updated.generation;
        this.nextCheck = updated.checkedAt + this.options.ttlMs;
        await this.prune([updated.generation, ...(state ? [state.generation] : [])]).catch(() => console.warn('Could not prune old result snapshots'));
        return candidate;
      } finally { if (!installed) await rm(directory, { recursive: true, force: true }); }
    } catch (error) {
      this.nextCheck = this.now() + Math.max(this.options.ttlMs, error instanceof HubRequestError ? error.retryAfterMs : 0);
      if (!this.current) throw error;
      console.error('Results refresh failed; retaining the validated snapshot:', error instanceof Error ? error.message : 'unknown error');
      this.current = { ...this.current, resultsSource: { ...this.current.resultsSource!, refreshFailed: true } };
      return this.current;
    } finally { await release?.(); }
  }
}
