import { modelName } from '../lib/comparison';
import type { ModelInfo } from '../lib/types';

/** A single display name, linked to the authored model page when available. */
export function ModelName({ model }: { model: ModelInfo }) {
  const url = model.hf_url || model.url;
  const name = modelName(model);
  return url
    ? <a href={url} target="_blank" rel="noreferrer" className="font-medium break-words underline-offset-4 hover:underline">{name}</a>
    : <span className="font-medium break-words">{name}</span>;
}
