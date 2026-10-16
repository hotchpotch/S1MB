"""Build an English Dataset PR body from validated results and author commentary."""

import json
from pathlib import Path

from .data import Result, read_json


def render_template(repository: Path, category: str, explanation: str | None) -> str:
    """Read provenance only after the caller has validated the repository."""
    lines = [
        "## Submission evidence",
        "",
        f"Evaluation category: `{category}`.",
        "",
        (
            "Leaderboard submission requires every active benchmark in the selected scope (Box is optional and excluded from ranking) "
            "complete and measured, across Choice, Noul and Score, with zero failed decisions. "
            "Missing benchmarks, unfinished tasks and partial results are not leaderboard-eligible. "
            "Validation success alone does not establish complete coverage. "
            "For input, capacity or API failures, follow the "
            "[adapter failure guide]"
            "(https://github.com/hotchpotch/S1MB/blob/main/docs/evaluation_failures.md) "
            "and complete a corrected evaluation before submitting."
        ),
    ]
    for folder in sorted(repository.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        results = []
        for path in sorted(folder.glob("*.json.xz")):
            result = Result.model_validate(read_json(path))
            result.predictions = []
            results.append(result)
        lines += [
            "",
            f"### `{folder.name}`",
            "",
            "| Task | Saved benchmarks | Complete | Successful decisions | Abstentions counted as incorrect | Failed decisions |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for task in ("choice", "noul", "score", "box"):
            rows = [r for r in results if r.benchmark.task == task]
            lines.append(
                f"| {task} | {len(rows)} | {sum(r.status == 'complete' for r in rows)} | {sum(r.counts.succeeded for r in rows)} | {sum(r.counts.abstained for r in rows)} | {sum(r.counts.failed for r in rows)} |"
            )
        # Deduplicate whole provenance objects, preserving mixed-run configurations.
        provenance = {}
        for result in results:
            record = result.model_dump(
                mode="json", exclude={"predictions", "metrics", "benchmark", "counts", "status"}
            )
            # Retain execution provenance without copying per-benchmark fields.
            record = {
                k: v
                for k, v in record.items()
                if k in {"model", "evaluator_version", "environment", "run_id", "provenance"}
            }
            record["environment"] = {
                k: v for k, v in result.environment.items() if k != "input_hashes"
            }
            key = json.dumps(record, sort_keys=True)
            provenance[key] = record
        lines += [
            "",
            "Recorded provenance and effective settings (distinct configurations):",
            "",
            "```json",
            json.dumps(list(provenance.values()), indent=2, ensure_ascii=False),
            "```",
        ]
    lines += [
        "",
        "## Training source and domain disclosure",
        "",
        (
            "For English v1 and Japanese v1, please disclose, to the best of your knowledge, "
            "whether the submitted model was trained on any original evaluation source family, "
            "directly or through mirrors, derivatives or mixtures, as well as the "
            "[Bekko System One Dataset](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0), "
            "the [Saihai Dataset](https://huggingface.co/datasets/hotchpotch/saihai-v1-dataset), "
            "or other datasets covering the same tasks or source domains as these evaluations."
        ),
        "",
        (
            "Review the [English/Japanese original dataset and source checklist]"
            "(https://github.com/hotchpotch/S1MB/blob/main/docs/evaluation_sources.md), "
            "which maps active benchmark IDs to HF dataset names and sources outside HF. "
            "English examples include `PolyAI/banking77`, `clinc/clinc_oos`, "
            "`fancyzhx/dbpedia_14`, `Anthropic/hh-rlhf` and `LocalLLaMA/typed-decisions`; "
            "Japanese examples include JGLUE, JSICK, `sbintuitions/JamC-QA`, "
            "`openai/MMMLU` and `hakari-bench/NanoJMTEB-v2`. "
            "These examples do not replace the full checklist. "
            "Not using Bekko System One or Saihai does not establish separation from "
            "S1MB source families or evaluation inputs."
        ),
        "",
        (
            "Identify the source families actually used, the affected S1MB benchmarks, "
            "and the training revisions, configurations, splits and filtering when known. "
            "Link model cards, training manifests, code or technical reports supporting the disclosure. "
            "Record supported benchmark-level declarations in model metadata as `training_exposure`, "
            "distinguishing `same_source_family`, `related_domain` and `evaluation_examples`. "
            "Reconcile audit findings with metadata and prose: distinguish confirmed evaluation-input "
            "use from approximate-match candidates, and do not claim no evaluation-input use "
            "when confirmed matches are reported. "
            "Training on an NLI dataset, for example, does not by itself establish exposure "
            "to every NLI benchmark. Filtering evaluation duplicates does not remove source-family exposure."
        ),
        "",
        (
            "If source details cannot be confirmed, state what is unknown. "
            "Missing disclosure does not mean a domain was unseen: other models may also have trained "
            "on these domains without public confirmation. Known-domain marks communicate documented "
            "training exposure, not an explanation or prediction of high scores; scores are unchanged."
        ),
        "",
        "## Author explanation and assessment",
        "",
        explanation.strip()
        if explanation
        else """[Replace this section before submitting. Generated evidence alone is not a complete PR description.]

- Explain additions, replacements, canonical model identity and aliases.
- State checkpoint availability, release plans and the corresponding adapter PR.
- Give exact smoke/full commands, runtime conditions and source revisions.
- Explain retries, actual failing limits/statuses, supported remedies, replacement run IDs and validation limitations; confirm complete coverage and zero failed decisions.
- List original source families reviewed, actual use, matching benchmark metadata and unknown training inventories.
- Disclose known training/evaluation overlap and whether settings were selected independently of test scores.
- Interpret the results in your own words, including strengths, weaknesses and uncertainty.
- Pin comparison/reference and PR parent commits; explain any remaining publication steps.""",
    ]
    return "\n".join(lines) + "\n"
