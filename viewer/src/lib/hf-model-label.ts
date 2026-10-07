/** Identify a Hub model and optional checkpoint directory from its explicit URL. */
export function hfModelLabel(raw: string | null | undefined): string | null {
  if (!raw) return null;
  try {
    const url = new URL(raw);
    if (url.hostname !== 'huggingface.co' || !['https:', 'http:'].includes(url.protocol)) return null;
    const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
    if (parts.length < 2 || ['datasets', 'spaces'].includes(parts[0])) return null;
    const directory = parts[2] === 'tree' ? parts.slice(4) : [];
    return [...parts.slice(0, 2), ...directory].join('/');
  } catch {
    return null;
  }
}
