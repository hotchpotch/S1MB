"""Language unions preserve separate provenance and do not require private data."""
import json

import pytest

from s1mb import cli, dataset_source
from s1mb.data import DATA_DIR, TrainingExposure, load_benchmark, load_category
from s1mb.result_repository import ModelMetadata


def test_all_refreshes_both_once_and_offline_skips(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(dataset_source, 'ensure_dataset', lambda root, **kwargs: events.append(kwargs.get('category', 'english-v1')))
    monkeypatch.setattr(cli, 'execute', lambda args, parser: events.append(args.category))
    command = ['s1mb', '--data-dir', str(tmp_path / 'data'), 'run', '--all', '--adapter', 'dummy']
    monkeypatch.setattr('sys.argv', command)
    cli.main()
    assert events == ['english-v1', 'japanese-v1', 'all']
    events.clear()
    monkeypatch.setattr('sys.argv', [*command, '--offline-dataset'])
    cli.main()
    assert events == ['all']
    monkeypatch.setattr('sys.argv', [*command, '--category', 'japanese-v1'])
    with pytest.raises(SystemExit):
        cli.main()


def test_japanese_active_inventory_is_ordinary_and_disjoint():
    japanese = load_category(DATA_DIR, 'japanese-v1')
    english = load_category(DATA_DIR, 'english-v1')
    assert len(japanese.benchmarks) == 27
    assert not set(japanese.benchmarks) & set(english.benchmarks)
    benchmarks = [load_benchmark(DATA_DIR, name) for name in japanese.benchmarks]
    assert sum(b.decision_count for b in benchmarks) == 8067
    assert all(b.evaluation_family == 'ordinary' for b in benchmarks)


def test_bekko_registry_is_bounded_pinned_and_explicit():
    registry = json.loads((DATA_DIR.parent / 'model-training-exposure.json').read_text())
    assert len(registry['model_ids']) == 3
    entries = [TrainingExposure.model_validate(e) for e in registry['training_exposure']]
    assert len(entries) == 98
    assert len({e.config for e in entries}) == 77
    assert {e.relationship for e in entries} == {'same_source_family'}
    assert {e.benchmark_id for e in entries} <= set(load_category(DATA_DIR, 'english-v1').benchmarks)
    for model_id in registry['model_ids']:
        metadata = ModelMetadata(model_id=model_id, display_name=model_id, short_name=model_id, training_exposure=entries)
        assert ModelMetadata.model_validate(metadata.model_dump()) == metadata
    assert ModelMetadata(model_id='owner__unknown', display_name='Unknown', short_name='Unknown').training_exposure is None


def test_all_evaluates_union_with_one_adapter_and_one_run_id(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from s1mb import runner

    seen = []

    def evaluate(adapter, root, benchmark, output, run_id, limit):
        seen.append((id(adapter), benchmark.id, run_id))
        return SimpleNamespace(counts=SimpleNamespace(failed=0), status='complete', metrics={}, elapsed_seconds=0, environment={})

    monkeypatch.setattr(runner, 'evaluate', evaluate)
    monkeypatch.setattr('sys.argv', ['s1mb', 'run', '--all', '--offline-dataset', '--adapter', 'dummy', '--output', str(tmp_path), '--run-id', 'synthetic-union'])
    cli.main()
    english = set(load_category(DATA_DIR, 'english-v1').benchmarks)
    japanese = set(load_category(DATA_DIR, 'japanese-v1').benchmarks)
    assert {bid for _, bid, _ in seen} == english | japanese
    assert len(seen) == len(english | japanese)
    assert len({adapter for adapter, _, _ in seen}) == 1
    assert {run_id for _, _, run_id in seen} == {'synthetic-union'}
