import { ArrowUpRight } from 'lucide-react';
import { GitHubIcon } from './GitHubIcon';
import { benchmarkSources } from '../lib/benchmark-sources';

export function BenchmarkSources({ dataset }: { dataset: string }) {
  const sources = benchmarkSources(dataset);
  if (!sources.length) return null;
  return <div aria-label="Benchmark sources" className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
    {sources.map(url => {
      const parsed = new URL(url);
      const label = parsed.pathname.replace(/^\/datasets\//, '').replace(/^\//, '');
      const provider = parsed.hostname === 'github.com' ? 'GitHub' : parsed.hostname === 'huggingface.co' ? 'Hugging Face' : parsed.hostname;
      return <a key={url} href={url} target="_blank" rel="noopener noreferrer" aria-label={`${provider}: ${label}`} className="inline-flex min-w-0 items-center gap-1.5 hover:text-foreground">
        {provider === 'GitHub' && <GitHubIcon />}
        {provider === 'Hugging Face' && <span aria-hidden="true" className="shrink-0">🤗</span>}
        <span className="break-all underline decoration-muted-foreground/40 underline-offset-4">{label}</span><ArrowUpRight aria-hidden="true" className="size-3 shrink-0" />
      </a>;
    })}
  </div>;
}
