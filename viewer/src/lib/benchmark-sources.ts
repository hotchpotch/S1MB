import sources from './benchmark-sources.json';

// Public source URLs curated from the dataset release's sources.json metadata.
// Keep only dataset identifiers and public links here, never row provenance or
// acquisition paths. AQuA's original repository is also linked by its HF card.
export function benchmarkSources(dataset: string): string[] {
  return (sources as Record<string, string[]>)[dataset] ?? [];
}
