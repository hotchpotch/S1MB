/** Show the repository and optional path for Hugging Face model links. */
export function ModelWebsiteLink({ url }: { url: string }) {
  let repository: string | null = null;
  try {
    const parsed = new URL(url);
    if (['huggingface.co', 'www.huggingface.co'].includes(parsed.hostname)) {
      repository = decodeURI(parsed.pathname).replace(/^\/+|\/+$/g, '') || 'Hugging Face';
    }
  } catch {
    // Keep the original link label if a URL cannot be parsed or decoded.
  }
  return <a href={url} target="_blank" rel="noreferrer" className="inline-flex min-w-0 items-start gap-1.5 underline decoration-muted-foreground/40 underline-offset-4 hover:text-primary" aria-label={repository ? `Hugging Face: ${repository}` : undefined}>
    {repository && <span aria-hidden="true" className="shrink-0 no-underline">🤗</span>}
    <span className="[overflow-wrap:anywhere]">{repository ?? 'Model website'}</span>
  </a>;
}
