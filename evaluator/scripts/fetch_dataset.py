"""Optional explicit dataset refresh; normal s1mb run refreshes automatically."""

import argparse
from pathlib import Path

from s1mb.data import DATA_DIR
from s1mb.dataset_source import dataset_session

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="english-v1")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    with dataset_session(args.data_dir, category=args.category):
        pass
