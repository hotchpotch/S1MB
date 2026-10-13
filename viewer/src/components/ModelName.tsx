import { TrainingBadge } from './TrainingExposure';
import { modelName } from '../lib/comparison';
import type { ModelInfo } from '../lib/types';

/** A single display name, linked to the authored model page when available. */
export function ModelName({ model, onClick, benchmarkIds }: { model: ModelInfo; onClick?: () => void; benchmarkIds?: string[] }) {
  const url = model.hf_url || model.url;
  const name = modelName(model);
  const content = onClick ? <button onClick={onClick} className="font-medium break-words underline-offset-4 hover:underline text-left">{name}</button> : url
    ? <a href={url} target="_blank" rel="noreferrer" className="font-medium break-words underline-offset-4 hover:underline">{name}</a>
    : <span className="font-medium break-words">{name}</span>;
  return <div className="flex items-start gap-1.5">{content}<span className="flex h-[1lh] shrink-0 items-center"><TrainingBadge model={model} benchmarkIds={benchmarkIds} onClick={onClick} /></span></div>;
}
