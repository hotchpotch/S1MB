/** Resolve the prepared display file before starting Next. */
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { networkInterfaces } from 'node:os';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const env = { ...process.env };
let dev = false, space = false;
let port = '3000';
let host = '127.0.0.1';
for (const address of networkInterfaces().tailscale0 ?? []) if (address.family === 'IPv4') host = address.address;
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === '--dev') dev = true;
  else if (arg === '--space') space = true;
  else if (['--display-file', '--port', '--host'].includes(arg) && args[i + 1] && !args[i + 1].startsWith('--')) {
    const value = args[++i];
    if (arg === '--display-file') env.S1MB_DISPLAY_FILE = path.resolve(value);
    else if (arg === '--port') port = value;
    else host = value;
  } else throw new Error(`Unknown or incomplete option: ${arg}`);
}
if (space) {
  if (!env.SPACE_ID) throw new Error('--space requires the managed HF Space environment');
  host = '0.0.0.0'; port = '7860';
}
const allowed = ['127.0.0.1', ...(networkInterfaces().tailscale0 ?? []).filter(a => a.family === 'IPv4').map(a => a.address)];
if (!space && !allowed.includes(host)) throw new Error('Bind to localhost or this machine’s Tailscale IPv4 address.');
const displayFile = path.resolve(env.S1MB_DISPLAY_FILE || path.join(root, 'display/viewer-summary.json'));
const child = spawn(process.execPath, [path.join(root, 'node_modules/next/dist/bin/next'), dev ? 'dev' : 'start', '--hostname', host, '--port', port], {
  cwd: root, stdio: 'inherit', env: { ...env, S1MB_DISPLAY_FILE: displayFile, NEXT_TELEMETRY_DISABLED: '1' },
});
for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => child.kill(signal));
child.on('exit', code => { process.exitCode = code ?? 1; });
