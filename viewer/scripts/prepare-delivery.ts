import path from 'node:path';
import { prepareDelivery } from '../src/lib/prepare-delivery';

const args = process.argv.slice(2);
if (args.length > 1) throw new Error('Usage: npm run prepare-delivery -- [DISPLAY_FILE]');
const start = performance.now();
const result = await prepareDelivery(path.resolve(args[0] ?? 'display/viewer-summary.json'));
console.log(JSON.stringify({ ...result, preparationMs: Math.round(performance.now() - start) }));
