/** Resolve runtime data before starting Next; no external files are needed at build time. */
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { networkInterfaces } from 'node:os';
import { hubResultsOptions } from '../src/lib/hub-results';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const dirs: string[] = [];
let dev = false;
let port = '3000';
let host = '127.0.0.1';
for (const address of networkInterfaces().tailscale0 ?? []) if (address.family === 'IPv4') host = address.address;
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === '--dev') dev = true;
  else if (arg === '--results-dir' && args[i + 1]) dirs.push(path.resolve(args[++i]));
  else if (arg === '--port' && args[i + 1]) port = args[++i];
  else if (arg === '--host' && args[i + 1]) host = args[++i];
  else throw new Error(`Unknown or incomplete option: ${arg}`);
}
const allowed = ['127.0.0.1', ...(networkInterfaces().tailscale0 ?? []).filter(a => a.family === 'IPv4').map(a => a.address)];
if (!allowed.includes(host)) throw new Error('Bind to localhost or this machine’s Tailscale IPv4 address.');
const dataDir = path.resolve(process.env.S1MB_DATA_DIR || path.join(root, 'data'));
const hub = hubResultsOptions();
if (hub && (dirs.length || process.env.S1MB_RESULTS_DIRS)) throw new Error('Choose either Hub results or --results-dir');
if (!hub && !dirs.length) dirs.push(path.join(dataDir, existsSync(path.join(dataDir, 'hub-results')) ? 'hub-results' : 'results'));
const child = spawn(process.execPath, [path.join(root, 'node_modules/next/dist/bin/next'), dev ? 'dev' : 'start', '--hostname', host, '--port', port], {
  cwd: root, stdio: 'inherit', env: { ...process.env, S1MB_DATA_DIR: dataDir, S1MB_RESULTS_DIRS: hub ? undefined : JSON.stringify(dirs), NEXT_TELEMETRY_DISABLED: '1' },
});
for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => child.kill(signal));
child.on('exit', code => { process.exitCode = code ?? 1; });
