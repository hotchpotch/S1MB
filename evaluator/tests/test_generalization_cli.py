from pathlib import Path
from types import SimpleNamespace

import pytest

from s1mb import cli


@pytest.mark.parametrize("task,expected", [(None, 6), ("noul", 2)])
def test_generalization_selection(monkeypatch, tmp_path, task, expected):
    selected = []

    def evaluate(adapter, data_dir, benchmark, output, run_id, limit):
        selected.append(benchmark)
        return SimpleNamespace(
            counts=SimpleNamespace(failed=0),
            status="complete",
            metrics={},
            elapsed_seconds=0,
            environment={},
        )

    monkeypatch.setattr("s1mb.runner.evaluate", evaluate)
    data = Path(__file__).resolve().parents[1] / "data"
    args = [
        "s1mb",
        "--data-dir",
        str(data),
        "run",
        "--adapter",
        "dummy",
        "--offline-dataset",
        "--generalization-only",
        "--output",
        str(tmp_path),
    ]
    if task:
        args += ["--task", task]
    monkeypatch.setattr("sys.argv", args)
    cli.main()
    assert len(selected) == expected
    assert all(b.dataset.startswith("datasets/s1mb-generalization-") for b in selected)
    if task:
        assert all(b.task == task for b in selected)


def test_generalization_rejects_empty_intersection(monkeypatch, capsys):
    data = Path(__file__).resolve().parents[1] / "data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "s1mb",
            "--data-dir",
            str(data),
            "run",
            "--adapter",
            "dummy",
            "--offline-dataset",
            "--generalization-only",
            "--benchmark",
            "arc-choice-test-v1",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert "No benchmarks match" in capsys.readouterr().err
