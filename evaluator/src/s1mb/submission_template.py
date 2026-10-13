"""Build an English Dataset PR body from validated results and author commentary."""

import json
from pathlib import Path

from .data import Result, read_json


def render_template(repository: Path, category: str, explanation: str | None) -> str:
    """Read provenance only after the caller has validated the repository."""
    lines = ["## Submission evidence", "", f"Evaluation category: `{category}`."]
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
            "| Task | Saved benchmarks | Complete | Successful decisions | Failed decisions |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for task in ("choice", "noul", "score"):
            rows = [r for r in results if r.benchmark.task == task]
            lines.append(
                f"| {task} | {len(rows)} | {sum(r.status == 'complete' for r in rows)} | {sum(r.counts.succeeded for r in rows)} | {sum(r.counts.failed for r in rows)} |"
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
        "## Author explanation and assessment",
        "",
        explanation.strip()
        if explanation
        else """[Replace this section before submitting. Generated evidence alone is not a complete PR description.]

- Explain additions, replacements, canonical model identity and aliases.
- State checkpoint availability, release plans and the corresponding adapter PR.
- Give exact smoke/full commands, runtime conditions and source revisions.
- Explain retries, changed settings, incomplete coverage and validation limitations.
- Disclose known training/evaluation overlap and whether settings were selected independently of test scores.
- Interpret the results in your own words, including strengths, weaknesses and uncertainty.
- Pin comparison/reference and PR parent commits; explain any remaining publication steps.""",
    ]
    return "\n".join(lines) + "\n"
