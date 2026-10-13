/** Authored disclosures only: absence never establishes an unseen domain. */
import { z } from 'zod';
import type { ModelInfo } from './types';
export const trainingExposureSchema = z.object({
  benchmark_id: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  relationship: z.enum(['same_source_family', 'related_domain', 'evaluation_examples']),
  dataset: z.string().max(256).regex(/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/).nullish(),
  revision: z.string().regex(/^[0-9a-f]{40}$/).nullish(),
  config: z.string().min(1).max(128).nullish(), split: z.string().min(1).max(64).nullish(),
  evidence_url: z.url().max(2048).refine(value => { const u = new URL(value); return ['http:', 'https:'].includes(u.protocol) && !u.username && !u.password; }),
  note: z.string().min(1).max(1000),
}).strict();
export const trainingExposuresSchema = z.array(trainingExposureSchema).min(1).max(256).nullish();
export type TrainingExposure = z.infer<typeof trainingExposureSchema>;
export function exposuresFor(model: ModelInfo, benchmarkId?: string): TrainingExposure[] {
  return (model.training_exposure ?? []).filter(e => benchmarkId == null || e.benchmark_id === benchmarkId);
}
export function exposureLabel(entries: TrainingExposure[]): string {
  return entries.some(e => e.relationship === 'evaluation_examples') ? 'Evaluation examples used' : 'Known domain';
}
export const exposureRelationship = { same_source_family: 'Same source family', related_domain: 'Related domain', evaluation_examples: 'Evaluation examples used' };
