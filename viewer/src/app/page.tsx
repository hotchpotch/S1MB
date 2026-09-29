import { Dashboard } from "../components/Dashboard";
import { getSnapshot } from "../lib/results";
export const dynamic = "force-dynamic";
export default async function Page({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  return (
    <Dashboard
      snapshot={await getSnapshot()}
      initialCategory={
        typeof params.category === "string" ? params.category : undefined
      }
      initialCheckedOnly={params.checkedOnly === "1"}
      initialModelSearch={typeof params.modelSearch === "string" ? params.modelSearch : undefined}
      initialGeneralizationOnly={params.generalOnly === "1"}
      initialTask={typeof params.task === "string" ? params.task : undefined}
      initialView={typeof params.view === "string" ? params.view : undefined}
      initialBenchmark={
        typeof params.benchmark === "string" ? params.benchmark : undefined
      }
      initialCompare={
        typeof params.compare === "string"
          ? [params.compare]
          : (params.compare ?? [])
      }
      initialRun={typeof params.run === "string" ? params.run : undefined}
    />
  );
}
