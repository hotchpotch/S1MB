import { LoaderCircle } from 'lucide-react';

export function LeaderboardLoading() {
  return (
    <main className="mx-auto min-h-screen max-w-6xl px-6 py-10 sm:px-10 sm:py-16">
      <div className="text-2xl font-bold tracking-tight">S1MB<span className="text-primary">.</span></div>
      <section className="mx-auto max-w-xl py-20 sm:py-28" aria-busy="true" aria-labelledby="loading-title">
        <div role="status" aria-live="polite">
          <LoaderCircle aria-hidden="true" className="mb-6 size-7 text-primary motion-safe:animate-spin" strokeWidth={1.5} />
          <h1 id="loading-title" className="text-3xl font-semibold tracking-tight sm:text-4xl">Preparing the leaderboard</h1>
          <p className="mt-4 text-muted-foreground leading-relaxed">Loading benchmark results. This can take a little longer after a restart.</p>
          <p className="mt-2 text-sm text-muted-foreground">Results will appear automatically. You can leave this page open.</p>
        </div>
        <div aria-hidden="true" className="mt-10 space-y-4 motion-safe:animate-pulse">
          {['w-4/5', 'w-3/5', 'w-2/3'].map(width => (
            <div key={width} className="flex items-center gap-5 rounded-lg border bg-card p-4">
              <div className="size-8 shrink-0 rounded bg-muted" />
              <div className={`h-3 rounded bg-muted ${width}`} />
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
