"""Optional explicit dataset refresh; normal s1mb run refreshes automatically."""

from s1mb.data import DATA_DIR
from s1mb.dataset_source import dataset_session

if __name__ == "__main__":
    with dataset_session(DATA_DIR):
        pass
