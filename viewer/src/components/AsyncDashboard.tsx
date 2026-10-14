'use client';

import { useEffect, useState } from 'react';
import { Dashboard } from './Dashboard';
import { LeaderboardLoading } from './LeaderboardLoading';
import { Button } from './ui/button';
import { loadDisplay } from '../lib/display-client';
import { parseDashboardState } from '../lib/dashboard-state';
import type { Snapshot } from '../lib/types';

export function AsyncDashboard({ load = loadDisplay }: { load?: () => Promise<Snapshot> }) {
  const [snapshot, setSnapshot] = useState<Snapshot>();
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    load().then(value => { if (active) setSnapshot(value); }, () => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [load, attempt]);
  if (snapshot) return <Dashboard snapshot={snapshot} initialState={parseDashboardState(new URLSearchParams(window.location.search))} />;
  if (failed) return <main className="mx-auto max-w-xl px-6 py-20">
    <h1 className="text-2xl font-semibold">Results could not be loaded</h1>
    <p role="alert" className="mt-4 text-muted-foreground">Please retry loading the leaderboard.</p>
    <Button className="mt-6" onClick={() => { setFailed(false); setAttempt(value => value + 1); }}>Retry</Button>
  </main>;
  return <LeaderboardLoading />;
}
