/** Offline conversion, never invoked by a web request. */
import { parseArgs } from 'node:util';
import path from 'node:path';
import { FilesystemResults } from '../src/lib/result-loader';
import { atomicJson, encodeDisplay, readDisplay, reductionWarnings } from '../src/lib/display-data';
const { values } = parseArgs({ options: {
  'results-dir': { type: 'string', multiple: true }, 'data-dir': { type: 'string', default: 'data' },
  output: { type: 'string', default: 'display/viewer-summary.json' }, previous: { type: 'string' },
  'source-repo': { type: 'string' }, 'source-revision': { type: 'string' }, 'approve-reduction': { type: 'string' },
} });
if (!values['results-dir']?.length) throw new Error('Provide --results-dir (repeatable)');
if (Boolean(values['source-repo']) !== Boolean(values['source-revision'])) throw new Error('Provide both source repo and revision');
const snapshot = await new FilesystemResults(path.resolve(values['data-dir']), values['results-dir'].map(d => path.resolve(d))).refresh();
const source = values['source-repo'] ? { repo: values['source-repo'], revision: values['source-revision']! } : null;
const artifact = encodeDisplay(snapshot, source);
const previous = values.previous ? await readDisplay(values.previous) : null;
const warnings = previous ? reductionWarnings(previous.snapshot, snapshot) : [];
const counts = (s: typeof snapshot) => ({ models: new Set(s.results.map(r => r.run_id)).size, results: s.results.length, benchmarks: s.benchmarks.length, completeResults: s.results.filter(r => r.status === 'complete').length });
const report = { source, digest: artifact.digest, previous: previous ? counts(previous.snapshot) : null,
  current: counts(snapshot), warnings, firstPublication: !previous };
await atomicJson(values.output + '.report.json', report);
console.log(JSON.stringify(report, null, 2));
if (warnings.length && values['approve-reduction'] !== artifact.digest) {
  console.error(`WARNING: publication blocked. Review the report, then explicitly approve this candidate with --approve-reduction ${artifact.digest}`);
  process.exitCode = 2;
} else {
  await atomicJson(values.output, artifact);
  console.log(`Generated ${values.output}`);
}
