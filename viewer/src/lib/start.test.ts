import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

test('start wrapper forwards CLI overrides and rejects incompatible options before launching Next', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 's1mb-start-'));
  const root = fileURLToPath(new URL('../../', import.meta.url));
  const preload = path.join(directory, 'capture.mjs');
  try {
    // Capture the real wrapper's child environment without starting an HTTP server.
    await writeFile(preload, `
      import cp from 'node:child_process';
      import { syncBuiltinESMExports } from 'node:module';
      import { EventEmitter } from 'node:events';
      cp.spawn = (_command, _args, options) => {
        console.log(JSON.stringify({ repo: options.env.S1MB_HF_RESULTS_REPO,
          revision: options.env.S1MB_HF_RESULTS_REVISION,
          dirs: options.env.S1MB_RESULTS_DIRS }));
        return new EventEmitter();
      };
      syncBuiltinESMExports();
    `);
    const run = (args: string[], repo = 'example/environment') => spawnSync(process.execPath,
      ['--import', 'tsx', '--import', pathToFileURL(preload).href, 'scripts/start.ts', ...args], {
        cwd: root, encoding: 'utf8', timeout: 10_000,
        env: { ...process.env, HF_TOKEN: '', S1MB_HF_RESULTS_REPO: repo,
          S1MB_HF_RESULTS_REVISION: 'main', S1MB_HF_RESULTS_CACHE_SECONDS: '3600', S1MB_RESULTS_DIRS: '' },
      });
    const inherited = run([]);
    assert.equal(inherited.status, 0, inherited.stderr);
    assert.deepEqual(JSON.parse(inherited.stdout), { repo: 'example/environment', revision: 'main' });
    const override = run(['--results-repo', 'example/cli', '--results-revision', 'refs/pr/7']);
    assert.equal(override.status, 0, override.stderr);
    assert.deepEqual(JSON.parse(override.stdout), { repo: 'example/cli', revision: 'refs/pr/7' });
    for (const [args, repo] of [
      [['--results-repo', 'example/cli', '--results-dir', './data/results'], ''],
      [['--results-repo', 'invalid'], 'example/environment'],
      [['--results-revision', 'main'], ''],
      [['--results-repo'], ''],
    ] as [string[], string][]) {
      const rejected = run(args, repo);
      assert.notEqual(rejected.status, 0);
      assert.equal(rejected.stdout, '', 'Invalid options must not start Next');
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});
