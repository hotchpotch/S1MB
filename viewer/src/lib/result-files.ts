/** Bounded server-side decoding of submitted JSON/XZ files. Requires xz-utils. */
import { execFile } from 'node:child_process';
import { readFile, stat } from 'node:fs/promises';
import { promisify } from 'node:util';
import { z } from 'zod';

const execute = promisify(execFile);
const limit = 64 * 1024 * 1024;
const count = z.number().int().nonnegative();
const link = z.url().refine(value => ['http:', 'https:'].includes(new URL(value).protocol)).nullable().optional();
export const metadataSchema = z.object({
  model_id: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]*__[A-Za-z0-9][A-Za-z0-9._-]*$/),
  display_name: z.string().min(1), short_name: z.string().min(1), url: link, hf_url: link,
  total_params: count.nullish(), active_params: count.nullish(),
  parameter_count_method: z.literal('non_lookup_parameters_v1').nullish(),
}).strict().refine(m => m.active_params == null ||
  (m.parameter_count_method === 'non_lookup_parameters_v1' && (m.total_params == null || m.active_params <= m.total_params)), 'Invalid parameter counts');

export async function readResultJson(file: string): Promise<unknown> {
  if (file.endsWith('.json.xz')) {
    const { stdout } = await execute('xz', ['--decompress', '--stdout', '--memlimit-decompress=128MiB', '--', file],
      { maxBuffer: limit, timeout: 30_000, encoding: 'utf8' });
    return JSON.parse(stdout);
  }
  if ((await stat(file)).size > limit) throw new Error('JSON exceeds 64 MiB');
  return JSON.parse(await readFile(file, 'utf8'));
}
