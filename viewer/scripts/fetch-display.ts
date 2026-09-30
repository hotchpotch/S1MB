/** Download only the public display artifact at an immutable Dataset revision. */
import { parseArgs } from 'node:util';
import { atomicJson, decodeDisplay, MAX_BYTES, SUMMARY_FILE } from '../src/lib/display-data';
const { values } = parseArgs({ options: { repo: { type: 'string', default: 'hotchpotch/s1mb-result' }, revision: { type: 'string' }, output: { type: 'string', default: `display/${SUMMARY_FILE}` } } });
if (!/^[a-f0-9]{40}$/.test(values.revision ?? '') || !/^[\w.-]+\/[\w.-]+$/.test(values.repo)) throw new Error('Provide repo and exact Dataset commit SHA');
const response = await fetch(`https://huggingface.co/datasets/${values.repo}/resolve/${values.revision}/${SUMMARY_FILE}`, { signal: AbortSignal.timeout(120000) });
if (!response.ok || !response.body) throw new Error(`Display download failed: ${response.status}`);
const chunks: Uint8Array[] = []; let bytes = 0;
const reader = response.body.getReader();
while (true) { const { done, value } = await reader.read(); if (done) break; bytes += value.length; if (bytes > MAX_BYTES) { await reader.cancel(); throw new Error('Display download exceeds size limit'); } chunks.push(value); }
const { artifact } = decodeDisplay(JSON.parse(Buffer.concat(chunks).toString('utf8')));
if (artifact.source?.repo !== values.repo) throw new Error('Display source repository mismatch');
await atomicJson(values.output, artifact);
console.log(`Downloaded ${bytes} bytes at ${values.revision}; results source ${artifact.source.revision}`);
