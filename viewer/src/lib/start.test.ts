import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

test('wrapper configures local and managed filesystem modes and rejects invalid options', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 's1mb-start-'));
  const preload = path.join(directory, 'capture.mjs');
  try {
    await writeFile(preload, `import cp from 'node:child_process'; import { syncBuiltinESMExports } from 'node:module'; import { EventEmitter } from 'node:events';
      cp.spawn = (_command, args, options) => { console.log(JSON.stringify({ args, dirs: options.env.S1MB_RESULTS_DIRS, seconds: options.env.S1MB_RESULTS_CHECK_SECONDS })); return new EventEmitter(); }; syncBuiltinESMExports();`);
    const run = (args: string[], space = '') => spawnSync(process.execPath, ['--import', 'tsx', '--import', pathToFileURL(preload).href, 'scripts/start.ts', ...args], {
      encoding: 'utf8', timeout: 10000, env: { ...process.env, S1MB_RESULTS_DIR: '/tmp/results', S1MB_RESULTS_CHECK_SECONDS: undefined, SPACE_ID: space },
    });
    const local = run([]); assert.equal(local.status, 0, local.stderr); assert.equal(JSON.parse(local.stdout).seconds, '0');
    const custom = run(['--results-dir', '/tmp/custom', '--check-seconds', '42']); assert.equal(custom.status, 0, custom.stderr);
    assert.equal(JSON.parse(custom.stdout).dirs, '["/tmp/custom"]'); assert.equal(JSON.parse(custom.stdout).seconds, '42');
    const space = run(['--space'], 'example/space'); assert.equal(space.status, 0, space.stderr);
    assert.equal(JSON.parse(space.stdout).seconds, '3600'); assert.ok(JSON.parse(space.stdout).args.includes('7860'));
    for (const args of [['--space'], ['--host', '0.0.0.0'], ['--check-seconds', '-1'], ['--results-dir'], ['--results-repo', 'example/results']]) {
      const rejected = run(args); assert.notEqual(rejected.status, 0); assert.equal(rejected.stdout, '');
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});
