"""Training disclosures are bounded provenance and cannot assert unseen status."""

import pytest
from pydantic import ValidationError

from s1mb.result_repository import ModelMetadata

EXPOSURE = {
    "benchmark_id": "jnli-choice", "relationship": "same_source_family",
    "dataset": "example/train", "revision": "a" * 40, "config": "nli", "split": "train",
    "evidence_url": "https://example.org/provenance", "note": "Train split; example overlap not established.",
}
METADATA = {"model_id": "example__model", "display_name": "Example", "short_name": "Example"}


def test_disclosure_missing_is_unknown_and_valid_provenance_round_trips():
    assert ModelMetadata.model_validate(METADATA).training_exposure is None
    m = ModelMetadata.model_validate({**METADATA, "training_exposure": [EXPOSURE]})
    assert ModelMetadata.model_validate(m.model_dump()).training_exposure == m.training_exposure


@pytest.mark.parametrize("change", [
    {"revision": "main"}, {"relationship": "unseen"}, {"note": "x" * 1001},
    {"split": "x" * 65},
    {"evidence_url": "https://user:secret@example.org"}, {"evidence_url": "file:///tmp/data"},
    {"predictions": []},
    {"registration_source": "verified"}, {"dataset_access": "open_weights"},
])
def test_invalid_training_provenance(change):
    with pytest.raises(ValidationError):
        ModelMetadata.model_validate({**METADATA, "training_exposure": [{**EXPOSURE, **change}]})


def test_bound_disclosure_inventory():
    for entries in ([], [EXPOSURE] * 257):
        with pytest.raises(ValidationError):
            ModelMetadata.model_validate({**METADATA, "training_exposure": entries})


def test_repository_rejects_unknown_benchmark_mapping_before_loading_measurements(tmp_path):
    from s1mb.data import write_json
    from s1mb.result_repository import validate_repository

    folder = tmp_path / "results" / "example__model"
    folder.mkdir(parents=True)
    write_json(folder / "metadata.json", {**METADATA, "training_exposure": [EXPOSURE]})
    with pytest.raises(ValueError, match="Unknown training exposure benchmark"):
        validate_repository(tmp_path / "definitions", folder.parent)


def test_export_rejects_unknown_training_benchmark_before_writing(tmp_path):
    from s1mb.data import write_json
    from s1mb.result_repository import export_results

    metadata = tmp_path / "metadata.json"
    output = tmp_path / "submission"
    write_json(metadata, {**METADATA, "training_exposure": [EXPOSURE]})
    with pytest.raises(ValueError, match="Unknown training exposure benchmark"):
        export_results(tmp_path / "data", [], metadata, output)
    assert not output.exists()


@pytest.mark.parametrize("entries", [[], [EXPOSURE] * 257])
def test_export_validates_authored_registry_bounds_before_writing(tmp_path, entries):
    from s1mb.data import write_json
    from s1mb.result_repository import export_results

    metadata = tmp_path / "metadata.json"
    output = tmp_path / "submission"
    write_json(metadata, METADATA)
    write_json(tmp_path / "model-training-exposure.json", {
        "disclosures": [{"model_ids": [METADATA["model_id"]], "training_exposure": entries}],
    })
    with pytest.raises(ValidationError):
        export_results(tmp_path / "data", [], metadata, output)
    assert not output.exists()


def test_unknown_training_snapshot_and_registry_groups():
    from s1mb.result_repository import TrainingExposureRegistry

    unknown = {**EXPOSURE, "dataset": None, "revision": None, "config": None, "split": None}
    m = ModelMetadata.model_validate({**METADATA, "training_exposure": [unknown]})
    assert m.training_exposure is not None
    assert m.training_exposure[0].revision is None
    groups = [{"model_ids": ["example__one"], "training_exposure": [EXPOSURE]},
              {"model_ids": ["example__two"], "training_exposure": [unknown]}]
    registry = TrainingExposureRegistry.model_validate({"disclosures": groups})
    assert registry.disclosures[1].training_exposure[0].revision is None
    for value in ({"disclosures": []}, {"disclosures": groups * 129},
                  {"disclosures": groups + [groups[0]]},
                  {"disclosures": [{**groups[0], "model_ids": ["example__one"] * 2}]},
                  groups[0]):
        with pytest.raises(ValidationError):
            TrainingExposureRegistry.model_validate(value)


@pytest.mark.parametrize("source", [None, "unknown", "author_declaration", "contributor_review"])
@pytest.mark.parametrize("access", [None, "unknown", "public", "private"])
def test_registration_and_access_are_independent_optional_provenance(source, access):
    entry = {**EXPOSURE, "registration_source": source, "dataset_access": access}
    metadata = ModelMetadata.model_validate({**METADATA, "training_exposure": [entry]})
    assert metadata.training_exposure is not None
    assert metadata.training_exposure[0].registration_source == source
    assert metadata.training_exposure[0].dataset_access == access
    assert ModelMetadata.model_validate(metadata.model_dump()) == metadata


def test_authored_registry_records_contributor_registration_without_inventing_data_access():
    from pathlib import Path

    from s1mb.data import read_json
    from s1mb.result_repository import TrainingExposureRegistry

    registry = TrainingExposureRegistry.model_validate(read_json(Path(__file__).parents[1] / "model-training-exposure.json"))
    for group in registry.disclosures:
        for entry in group.training_exposure:
            assert entry.registration_source == "contributor_review"
            if "mohit67890__imajev-4b" in group.model_ids:
                assert entry.dataset_access == "unknown"
                assert entry.dataset is None and entry.revision is None
                continue
            if entry.dataset_access is not None:
                assert entry.dataset_access == "public"
                assert entry.dataset == "hotchpotch/bekko-system-one-dataset-v0"
                assert entry.revision == "c6a49c450fe88b40bd96c54d9cc19fafd9aa5591"


def test_export_matches_second_registry_group_and_explicit_metadata_wins(tmp_path):
    from s1mb.data import write_json
    from s1mb.result_repository import export_results

    metadata = tmp_path / "metadata.json"
    write_json(metadata, METADATA)
    write_json(tmp_path / "model-training-exposure.json", {"disclosures": [
        {"model_ids": ["other__model"], "training_exposure": [EXPOSURE]},
        {"model_ids": [METADATA["model_id"]], "training_exposure": [EXPOSURE]},
    ]})
    with pytest.raises(ValueError, match="Unknown training exposure benchmark"):
        export_results(tmp_path / "data", [], metadata, tmp_path / "out")
    write_json(metadata, {**METADATA, "training_exposure": [{**EXPOSURE, "benchmark_id": "explicit"}]})
    with pytest.raises(ValueError, match="Unknown training exposure benchmark: explicit"):
        export_results(tmp_path / "data", [], metadata, tmp_path / "out")
