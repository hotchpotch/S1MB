"""Read System One v1 judgments from Hugging Face save_to_disk folders."""

import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from .data import Case


def _text(value: Any) -> str:
    """Keep text verbatim; render structured instructions/criteria deterministically."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def case_from_row(row: dict[str, Any]) -> Case:
    source = row["input"]
    decisions = source["decisions"]
    ids = [d["id"] for d in decisions]
    targets_by_id = {t["decision_id"]: t for t in row["targets"]}
    if (
        len(set(ids)) != len(ids)
        or len(targets_by_id) != len(row["targets"])
        or set(ids) != set(targets_by_id)
    ):
        raise ValueError("Every evaluation decision must have exactly one target")

    questions, targets = [], {}
    for decision in decisions:
        key = decision["id"]
        if decision["kind"] != "judgment":
            raise ValueError("Ranking requires ranking metrics; S1MB evaluates judgments only")
        if decision.get("documents") or decision.get("scoring") is not None:
            raise ValueError("Judgments cannot contain ranking documents or relative scoring")
        criteria = decision["criteria"]
        candidates = {c["id"]: c for c in criteria}
        if len(candidates) != len(criteria):
            raise ValueError("Duplicate criterion IDs")
        order = [c["id"] for c in criteria]
        target = targets_by_id[key]
        if target["kind"] != "judgment_distribution":
            raise ValueError("Judgment requires a judgment_distribution target")
        if len(set(target["ids"])) != len(target["ids"]):
            raise ValueError("Duplicate target IDs")
        questions.append(
            {
                "id": key,
                "task": decision["type"],
                "instructions": _text(json.loads(decision["instructions_json"])),
                "instructions_json": decision["instructions_json"],
                "system_prompt": decision.get("system_prompt") or "",
                "options": [
                    {
                        "id": oid,
                        "description": _text(json.loads(candidates[oid]["description_json"])),
                        "description_json": candidates[oid]["description_json"],
                        "value": candidates[oid]["value"],
                    }
                    for oid in order
                ],
            }
        )
        targets[key] = {
            "kind": target["annotation_kind"],
            "probabilities": dict(zip(target["ids"], target["probabilities"], strict=True)),
            "metadata": json.loads(target.get("metadata_json") or "{}"),
        }
    return Case.model_validate(
        {
            "case_id": row["case_id"],
            "group_id": row["group_id"],
            "language": row["language"],
            "state": json.loads(source["state_json"]),
            "state_json": source["state_json"],
            "questions": questions,
            "targets": targets,
            "provenance": {
                "input_hash": row["input_hash"],
                "split": row["split"],
            },
        }
    )


def load_hf_cases(directory: Path, split: str) -> list[Case]:
    dataset = load_from_disk(str(directory))
    if not isinstance(dataset, DatasetDict) or split not in dataset:
        raise ValueError(f"Expected a saved DatasetDict containing split {split}")
    cases = []
    for row in dataset[split]:
        if row["split"] != split:
            raise ValueError("Row split differs from requested split")
        cases.append(case_from_row(row))
    return cases
