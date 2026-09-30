"""Build from latest public results with Xet, then optionally publish with a parent-SHA guard."""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download

REPO = "hotchpotch/s1mb-result"
SUMMARY = "viewer-summary.json"
VIEWER = Path(__file__).resolve().parents[1]


def result_paths(info):
    """Select actual model files only; never treat the display artifact as a result."""
    files = {f.rfilename: f for f in info.siblings}
    paths = sorted(name for name in files if name.endswith(".json.xz"))
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\.json\.xz", name) for name in paths):
        raise ValueError("Unexpected published result path")
    if not paths:
        raise ValueError("No published result files found")
    for folder in sorted({name.split("/")[0] for name in paths}):
        metadata = f"{folder}/metadata.json"
        if metadata not in files:
            raise ValueError(f"Missing {metadata}")
        paths.append(metadata)
    if len(paths) > 20000:
        raise ValueError("Too many result files")
    for name in paths:
        size = files[name].size
        if size is None or size > 64 * 1024 * 1024:
            raise ValueError(f"Invalid result size: {name}")
    return paths


def check_inventory(paths, candidate):
    """Every source measurement belonging to current definitions must survive conversion."""
    payload = candidate["payload"]
    definitions = {payload["benchmarks"][i]["id"] for i in payload["definitions"]}
    expected = {(name.split("/")[0], name.split("/")[1][:-8]) for name in paths
                if name.endswith(".json.xz") and name.split("/")[1][:-8] in definitions}
    actual = {(payload["models"][r["model"]]["run_id"], payload["benchmarks"][r["benchmark"]]["id"])
              for r in payload["results"]}
    if actual != expected or not actual:
        raise ValueError(f"Source inventory mismatch: expected {len(expected)}, generated {len(actual)}")
    print(f"Verified all {len(actual)} active source measurements", flush=True)


def publish(api, sha, output):
    """The Hub rejects a concurrent change instead of publishing stale data."""
    return api.create_commit(repo_id=REPO, repo_type="dataset", revision="main", parent_commit=sha,
        operations=[CommitOperationAdd(path_in_repo=SUMMARY, path_or_fileobj=str(output))],
        commit_message=f"Prepare leaderboard display from {sha}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=VIEWER / "display" / SUMMARY)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--approve-reduction", help="Reviewed candidate digest, printed when generation stops")
    args = parser.parse_args()
    output = args.output.resolve()
    # Public reads deliberately use no credentials. Only the final write uses local authentication.
    public = HfApi(token=False)
    info = public.dataset_info(REPO, revision="main", files_metadata=True)
    if not info.sha or not re.fullmatch(r"[a-f0-9]{40}", info.sha):
        raise ValueError("Cannot resolve latest results commit")
    paths = result_paths(info)
    print(f"Latest results: {REPO}@{info.sha}; {len(paths)} source files", flush=True)
    with tempfile.TemporaryDirectory(prefix="s1mb-display-") as temporary:
        root = Path(temporary)
        cache = Path(snapshot_download(REPO, repo_type="dataset", revision=info.sha,
                     allow_patterns=paths, token=False, max_workers=4))
        results = root / "results"
        for name in paths:
            destination = results / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache / name, destination)
        command = ["node", "--import", "tsx", "scripts/generate-display.ts", "--results-dir", str(results),
                   "--source-repo", REPO, "--source-revision", info.sha, "--output", str(output)]
        old = next((f for f in info.siblings if f.rfilename == SUMMARY), None)
        if old:
            if old.size is None or old.size > 32 * 1024 * 1024:
                raise ValueError("Previous display artifact exceeds size limit")
            previous = hf_hub_download(REPO, SUMMARY, repo_type="dataset", revision=info.sha, token=False)
            command += ["--previous", previous]
        if args.approve_reduction:
            command += ["--approve-reduction", args.approve_reduction]
        subprocess.run(command, cwd=VIEWER, check=True, timeout=900)
    candidate = json.loads(output.read_text())
    assert candidate["source"] == {"repo": REPO, "revision": info.sha}
    check_inventory(paths, candidate)
    print(f"Display JSON: {output.stat().st_size} bytes", flush=True)
    if args.publish:
        commit = publish(HfApi(), info.sha, output)
        print(f"Published display artifact: {REPO}@{commit.oid}", flush=True)
    else:
        print("Generated only. Use --publish to upload after review.", flush=True)


if __name__ == "__main__":
    main()
