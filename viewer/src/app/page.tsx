import { Dashboard } from "../components/Dashboard";
import { getSnapshot } from "../lib/results";
import { parseDashboardState } from "../lib/dashboard-state";
export const dynamic = "force-dynamic";
export default async function Page({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    for (const item of Array.isArray(value) ? value : value === undefined ? [] : [value]) query.append(key, item);
  }
  return <Dashboard snapshot={await getSnapshot()} initialState={parseDashboardState(query)} />;
}
