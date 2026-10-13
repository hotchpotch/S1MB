---
name: s1mb-pr-check
description: Review S1MB GitHub code and Hugging Face result PRs for security, evaluation integrity, duplicate models, and training disclosures; merge and update the leaderboard within the user's authorized scope.
---

# S1MB PR Check

Produce an evidence-backed acceptance decision for S1MB submissions and carry
explicitly authorized publication through to a verified leaderboard Space.

## Scope and working record

Read the repository's `AGENTS.md` first. Resolve project links below relative to
this skill directory. Refresh remote state; an earlier session's PR list, main
revision, evaluator schema, and deployment state may be stale.

Distinguish review, public comments, merge, summary publication, and deployment.
Invoking this skill alone authorizes none of the external writes. Honor existing
user authorization without asking again. Review-only requests end with a report;
approval of specific acceptable PRs excludes held PRs and unrelated new PRs.
Comments require authorization to post. Follow the project language rules and
thank contributors respectfully; mention the actual author when requested.

Keep a resumable record under ignored `tmp/`: requested scope, PR URLs/authors,
reviewed head/base SHAs, checks and limitations, decisions and authorization,
merged commits, Dataset source/published SHAs, and Actions/Space revisions.
Keep credentials and private evaluation inputs out of reports and tracked source.

## Establish the real changes

- List open GitHub PRs in `hotchpotch/S1MB`, including draft status, current heads,
  comments/reviews, files and CI. List open HF Dataset pull requests, including
  drafts, in `hotchpotch/s1mb-result` using Hub discussion APIs. Read the latest
  explanation and replies rather than relying on the original title.
- Pin downloaded evidence to immutable SHAs. Compare code with its actual base
  and results with their branch origin/merge base. Files added to main after
  branching are not deletions proposed by an old PR. Separately inspect conflicts
  and overlapping edits against current main.
- Apply canonical model identity and public-weight preference rules from
  [the submission guide](../../../docs/contributing_results.md). Compare with
  existing results and prior accepted submissions. Distinguish new trained
  releases from quantized copies, mirrors, API aliases and resubmissions; display
  names alone do not establish identity.
- Different trained models belong in separate result PRs. Variants of the same
  trained model may share a PR but retain one canonical model folder. Flag mixed
  distinct-model submissions for separation before acceptance; a comparison table
  may include reference models without uploading their files.
- If there are no open code PRs, say so. A result review or inspection of an
  existing adapter is not a security audit of all GitHub code.

## Code and security

Follow [the developer workflow](../../../docs/developer_workflow.md) for relevant
checks. Inspect submitted code before executing it. Use an isolated checkout and
credential-free environment for untrusted tests. Do not expose private data,
execute model-repository programs, load weights, or use a GPU merely for review.
Static inspection and synthetic tests usually suffice.

Trace external endpoints and user-controlled URLs/paths, credentials, remote-code
loading, unsafe deserialization, shell execution, file writes, input/decompression
bounds, batching/concurrency limits, timeouts/retries, and raw error responses.
Inspect target/identifier leakage into model text, overflow/truncation policies,
and probability handling for scoring changes.

Separate vulnerabilities introduced by source PRs from risks in optional external
serving code: result-only PRs do not install that code in the leaderboard. State
exploit conditions, affected component and severity. Local-only service assumptions
must be reconsidered before external exposure. Do not claim proof of security
from an absence of detected issues.

Report latest-head CI separately from local checks. Pending, approval-required,
skipped or stale-head CI does not establish current success. Run meaningful
checks for the change without repeating checks already sufficient for that head.

## Results and training disclosures

Read [the submission guide](../../../docs/contributing_results.md) and
[evaluation policy](../../../docs/evaluation.md). For known flags also read
[training disclosures](../../../docs/training_exposure.md).

- Download submission files at the reviewed SHA. Check allowed paths, inventory,
  file sizes and Hub hashes. Bound XZ/JSON decoding; inspect unexpected executable
  artifacts and secret patterns without printing sensitive values. Use pinned
  manifests/hashes for identity rather than downloading entire checkpoints.
- Run `s1mb validate-results` from `evaluator/` with the current trusted evaluator.
  Confirm the code/schema actually imported; an outdated local checkout may reject
  metadata accepted by current main. Validate against each result's recorded
  Dataset revision, preserving original measurements and provenance.
- Count required coverage, successful/failed decisions and incomplete results by
  scope. Incomplete results do not qualify for complete overall scores. Keep
  private Japanese rows and targets private during validation and reporting.
- Check exact model revision evidence, effective settings, reruns and selection
  history. A local revision label plus claimed equivalence to uploaded weights
  is weaker evidence than a recorded SHA. Hash and metric validation do not prove
  which model generated predictions.
- Look for selection or tuning on S1MB test labels/scores: temperature, prompts,
  checkpoints, or scoring rules. A fixed rule derived from a prior test search
  does not establish independent selection. Distinguish S1MB tuning from disclosed
  selection on other benchmarks, and report uncertainty without alleging misconduct.
  S1MB provides no validation split.
- Check TP/AP and `parameter_count_method`. Static AP is all parameters minus
  static embeddings, including tied output weights, not routed per-token MoE
  parameters. Unknown AP is not itself a blocker. Do not estimate it from names.
- Validate benchmark-level training declarations, relationship types, evidence
  URLs and the author's explanation. Same-source and related-domain declarations
  produce known-domain marks; evaluation-example use is distinct. Missing means
  unknown, not unseen. Supported author declarations are not independently proven
  training histories. Known flags never adjust scores.

Report PRs as acceptable, changes required, or held for a stated decision. Include
supporting checks, security findings, CI state and remaining uncertainty. Do not
invent new acceptance policy because a submission has unusual metadata.

## Authorized merge and leaderboard publication

Re-fetch each head, conflicts and required checks before merging. If the head
changed, review and validate its new delta. Use GitHub's expected-head guard where
available. HF's discussion merge API lacks that guard: recheck immediately before
merging and verify the resulting main commit/inventory. Preserve history and
existing files. Inspect concurrent changes before retrying rejected writes; never
blindly repeat a mutation. Leave held PRs untouched.

For authorized publication/deployment, follow
[the Space runbook](../../../docs/huggingface_space_deploy.md):

1. Generate from latest public results with `viewer/scripts/publish-display.py`,
   initially without `--publish`, to an ignored output directory. Review source
   SHA, inventory/completion comparison, parameter report, language scopes, and
   the submitted models' known flags. Unsupported counts may remain unknown.
2. Compare with the published summary. Reductions require human approval of the
   exact candidate digest; general merge/deploy approval does not cover unexpected
   reductions. Preserve visibly incomplete results.
3. Publish only the verified `viewer-summary.json` with its source Dataset SHA
   as the parent guard. Use the existing script or its `publish` helper for an
   already reviewed candidate. Revalidate changed candidates/concurrent updates.
   Do not incidentally rewrite measurements or publish metadata changes.
4. Inspect worktrees and preserve uncommitted work. Reconcile current `main` and
   `hf-space-docker` with the runbook's cross-merges/fast-forwards, never force
   pushes. Push or dispatch deployment after the summary has been published.
5. Wait for validation, image publication and deployment success. Inspect failed
   jobs before retrying and keep the user informed during waits.
6. Verify the exact running Space revision, pinned image/source commit, HTTP
   success and target rows/flags. Verify both scopes when both changed. Preserve
   visibility and configured `fullWidth: true` and `header: mini`. Workflow
   success alone does not establish live UI content.

Finish with PR/Actions/Space links, actions actually completed, verification and
material limitations. Distinguish stored model folders from complete leaderboard
rows when reporting counts.
