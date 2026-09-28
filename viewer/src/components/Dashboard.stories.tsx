import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Dashboard } from "./Dashboard";
import type { Snapshot, Task } from "../lib/types";
import { METRICS } from "../lib/types";
const benchmarks = (["choice", "noul", "score"] as Task[]).map((task) => ({
  id: `example-${task}-v1`,
  task,
  dataset: "datasets/example.jsonl",
  split: "test",
  case_count: 100,
  decision_count: 100,
  primary_metric: METRICS[task].name,
}));
const fixture: Snapshot = {
  categories: [
    {
      id: "english-v1",
      name: "English v1 · Mock",
      description:
        "Synthetic Storybook fixtures. These are not measured results.",
      benchmarks: benchmarks.map((b) => b.id),
    },
  ],
  benchmarks,
  scoring: Object.fromEntries(benchmarks.map(b => [b.id, { eligible: true, reason: null }])),
  sources: [{ name: "storybook-mocks", files: 3 }],
  issues: [],
  results: benchmarks.map((b) => ({
    run_id: "synthetic-demo",
    benchmark: b,
    model: {
      id: "Example System One model · Synthetic",
      adapter: "dummy",
      revision: "demo",
      settings: {},
    },
    created_at: "2026-09-25T00:00:00Z",
    evaluator_version: "0.1.0",
    provenance: "demo",
    status: "complete",
    counts: { cases: 100, expected: 100, succeeded: 100, failed: 0 },
    metrics: { [b.primary_metric]: b.task === "choice" ? 0.72 : 0.18,
      baseline_adjusted_score: 0.44, baseline_adjusted_skill: 0.44,
      ...(b.task === 'noul' ? { accuracy: 0.76, balanced_accuracy: 0.72, positive_recall: 0.64, specificity: 0.8 }
        : b.task === 'choice' ? { fixed_answer_accuracy_baseline: 0.5 }
        : { normalized_score_rmse: 0.22, constant_mae_baseline: 0.32 }),
    },
    elapsed_seconds: 3,
  })),
};
const meta = {
  title: "S1MB/Dashboard",
  component: Dashboard,
  args: { snapshot: fixture },
} satisfies Meta<typeof Dashboard>;
export default meta;
type Story = StoryObj<typeof meta>;
export const Default: Story = {};
export const Empty: Story = { args: { snapshot: { ...fixture, results: [] } } };
export const Partial: Story = {
  args: {
    snapshot: {
      ...fixture,
      results: fixture.results.map((r) => ({
        ...r,
        status: "partial",
        counts: { ...r.counts, cases: 3, succeeded: 3 },
      })),
    },
  },
};
export const Error: Story = {
  args: {
    snapshot: {
      ...fixture,
      results: [],
      issues: [
        "Invalid result: unsupported format version.",
        "Conflicting result excluded: synthetic-demo / example-choice-v1",
      ],
    },
  },
};
export const RunDetails: Story = { args: { initialRun: "synthetic-demo" } };
export const BatchVariants: Story = {
  args: {
    snapshot: {
      ...fixture,
      results: [
        ...fixture.results.map((r) => ({
          ...r,
          model: {
            ...r.model,
            settings: { renderer: "reviewed-system-and-instruction-v1" },
          },
        })),
        ...fixture.results.map((r) => ({
          ...r,
          run_id: "synthetic-question-batch",
          model: {
            ...r.model,
            settings: { questions_per_call: 1 },
          },
        })),
      ],
    },
  },
};

const comparisonFixture: Snapshot = {
  ...fixture,
  results: [
    ...fixture.results.map((r) => ({
      ...r,
      run_id: "mock-default",
      model: {
        ...r.model,
        settings: { renderer: "reviewed-system-and-instruction-v1" },
      },
    })),
    ...fixture.results
      .filter((r) => r.benchmark.task === "choice")
      .map((r) => ({
        ...r,
        run_id: "mock-choice-only",
        model: { ...r.model, id: "Choice-only model · Synthetic" },
      })),
    ...fixture.results.map((r) => ({
      ...r,
      run_id: "mock-partial",
      status: "partial" as const,
      counts: { ...r.counts, cases: 3, succeeded: 3 },
    })),
  ],
};
export const Benchmarks: Story = {
  args: { initialView: "benchmarks", snapshot: comparisonFixture },
};
export const Comparison: Story = {
  args: {
    initialView: "compare",
    initialCompare: ["mock-default", "mock-choice-only", "mock-partial"],
    snapshot: comparisonFixture,
  },
};
export const EmptyComparison: Story = { args: { initialView: "compare" } };

export const NoulDiagnostics: Story = { args: { initialView: "benchmarks", initialTask: "noul" } };

// Synthetic measurements exercise ranking and layout, not model performance.
const previewNames = [
  "Jev-style reference · Synthetic",
  "Very-long-checkpoint-name-without-spaces-bekko68-instruction_state-30pct · Synthetic",
  "Small encoder · Synthetic",
  "Small encoder, alternate instructions · Synthetic",
];
const layoutFixture: Snapshot = {
  ...fixture,
  results: previewNames.flatMap((name, i) => fixture.results.map(r => ({
    ...r,
    run_id: `synthetic-layout-${i}`,
    provenance: 'measured' as const,
    model: { ...r.model, id: name, adapter: 'storybook-fixture' },
    metrics: { ...r.metrics, baseline_adjusted_score: 0.7 - i * 0.12,
      f1: 0.8 - i * 0.1, precision: 0.85 - i * 0.1, positive_recall: 0.75 - i * 0.1 },
  }))),
};
export const LeaderboardLayout: Story = { args: { snapshot: layoutFixture } };
export const NoulMetrics: Story = { args: { snapshot: layoutFixture, initialView: 'benchmarks', initialTask: 'noul' } };
export const LongNameComparison: Story = { args: {
  snapshot: layoutFixture, initialView: 'compare', initialCompare: previewNames.map((_, i) => `synthetic-layout-${i}`),
} };
export const MobileComparison: Story = { ...LongNameComparison, globals: { viewport: { value: 'mobile1', isRotated: false } } };

export const ReleaseLabels: Story = { args: { snapshot: {
  ...layoutFixture,
  results: layoutFixture.results.map((r, i) => ({ ...r, model: {
    ...r.model, adapter: 'bekko',
    id: i < 3 ? 'example/bekko-large' : 'example/bekko-small',
    settings: { renderer: 'reviewed-prompt-v1', checkpoint_encoder_config: { lora: i < 3 ? { rank: 16 } : null } },
  } })),
} } };


export const UndefinedNoulMetric: Story = { args: {
  initialView: 'benchmarks', initialTask: 'noul',
  snapshot: { ...layoutFixture, results: layoutFixture.results.map((r,i) => i < 3 ? { ...r, metrics: { ...r.metrics, f1: null, precision: null } } : r) },
} };


const longComparison: Snapshot = {
  ...layoutFixture,
  benchmarks: Array.from({ length: 15 }, (_, i) => benchmarks.map(b => ({ ...b, id: `example-${i}-${b.task}-v1` }))).flat(),
  results: Array.from({ length: 15 }, (_, i) => layoutFixture.results.map(r => ({ ...r, benchmark: { ...r.benchmark, id: `example-${i}-${r.benchmark.task}-v1` } }))).flat(),
};
longComparison.categories = [{ ...fixture.categories[0], benchmarks: longComparison.benchmarks.map(b => b.id) }];
longComparison.scoring = Object.fromEntries(longComparison.benchmarks.map(b => [b.id, { eligible: true, reason: null }]));
export const ScrollableComparison: Story = { args: {
  snapshot: longComparison, initialView: 'compare', initialCompare: ['synthetic-layout-0', 'synthetic-layout-1', 'synthetic-layout-2'],
} };

export const FocusedLeaderboard: Story = { args: { snapshot: layoutFixture }, parameters: {
  docs: { description: { story: "Overall ranking is the default. Full-task and general-purpose scores are always visible; score bars share a fixed 0–100% scale." } },
} };

export const BenchmarkBrowsing: Story = { args: {
  snapshot: longComparison, initialView: 'benchmarks', initialTask: 'noul',
} };

export const ImbalancedNoul: Story = { args: {
  initialView: 'benchmarks', initialTask: 'noul',
  snapshot: { ...layoutFixture, results: layoutFixture.results.map(r => ({ ...r, metrics: {
    ...r.metrics, positive_prevalence: 0.11, majority_accuracy_baseline: 0.89,
  } })) },
} };

export const TaskProfiles: Story = { args: {
  initialView: 'compare', initialCompare: ['synthetic-layout-0', 'synthetic-layout-1'],
  snapshot: { ...layoutFixture, results: layoutFixture.results.map(r => ({ ...r, metrics: {
    ...r.metrics, baseline_adjusted_score: r.run_id === 'synthetic-layout-0' ? (r.benchmark.task === 'score' ? 0.15 : 0.8) : (r.benchmark.task === 'score' ? 0.85 : 0.4),
  } })) },
} };

export const DifferenceComparison: Story = { ...TaskProfiles, parameters: {
  docs: { description: { story: "Largest differences first within each task. Missing comparisons remain at the end; gap is descriptive, not statistical significance." } },
} };

export const ReferenceComparison: Story = { ...TaskProfiles, parameters: {
  docs: { description: { story: "Enable difference from reference, choose either model, and switch raw/adjusted scores. Positive delta always means better." } },
} };

export const ModelDetailPanel: Story = { args: {
  snapshot: longComparison, initialRun: 'synthetic-layout-0',
} };

export const MobileLeaderboard: Story = { ...FocusedLeaderboard, globals: { viewport: { value: 'mobile1', isRotated: false } } };
export const MobileTaskProfiles: Story = { ...TaskProfiles, globals: { viewport: { value: 'mobile1', isRotated: false } } };

export const RepeatedModelRuns: Story = { args: {
  snapshot: { ...layoutFixture, results: layoutFixture.results.map(r => ({ ...r, model: { ...r.model, id: 'Same checkpoint · Synthetic' } })) },
} };
export const RepeatedModelComparison: Story = { args: {
  ...RepeatedModelRuns.args, initialView: 'compare', initialCompare: ['synthetic-layout-0', 'synthetic-layout-1'],
} };
