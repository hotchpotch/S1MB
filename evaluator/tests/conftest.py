"""Keep public unit tests independent of downloaded evaluation data."""

import os

import pytest

from s1mb.data import DATA_DIR


def pytest_configure(config):
    config.addinivalue_line("markers", "dataset: requires the downloaded evaluation dataset")


def pytest_collection_modifyitems(items):
    available = (DATA_DIR / "datasets/hub-source.json").is_file()
    if available and not os.environ.get("S1MB_TEST_NO_DATASET"):
        return
    for item in items:
        if item.get_closest_marker("dataset"):
            item.add_marker(pytest.mark.skip(reason="Evaluation dataset is not installed"))
