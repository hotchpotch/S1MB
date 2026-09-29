"""Prepare evaluation revisions before replacing this process with Next.js."""

import argparse
import os
import re
from pathlib import Path

from s1mb.dataset_source import dataset_session
from s1mb.result_repository import sync_results


def prepare(root: Path, env: dict[str, str]) -> None:
    repo = env.get("S1MB_HF_RESULTS_REPO", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo):
        raise ValueError("Configure the results Dataset repository in the Space settings")
    if env.get("S1MB_RESULTS_DIRS"):
        raise ValueError("Space startup requires Hub results mode")
    if (env.get("S1MB_DATA_DIR") or str(root)) != str(root):
        raise ValueError("The Space image uses its prepared evaluator/data directory")
    revision = env.get("S1MB_HF_RESULTS_REVISION") or "main"
    # Current definitions need current inputs even when every result uses older data.
    with dataset_session(root):
        sync_results(root, repo, revision, root / "hub-results")


def runtime_environment(argv: list[str] | None, env: dict[str, str]) -> dict[str, str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-repo", help="Results Dataset ID (overrides the environment)")
    parser.add_argument("--results-revision", help="Results branch, tag, or commit")
    args = parser.parse_args(argv)
    configured = dict(env)
    if args.results_repo is not None:
        configured["S1MB_HF_RESULTS_REPO"] = args.results_repo
    if args.results_revision is not None:
        configured["S1MB_HF_RESULTS_REVISION"] = args.results_revision
    return configured


def main(argv: list[str] | None = None) -> None:
    configured = runtime_environment(argv, dict(os.environ))
    root = Path("/app/evaluator/data")
    os.chdir(root.parent)
    prepare(root, configured)
    env = {**configured, "S1MB_DATA_DIR": str(root), "NEXT_TELEMETRY_DISABLED": "1"}
    os.chdir("/app/viewer")
    os.execvpe("node", ["node", "node_modules/next/dist/bin/next", "start",
                         "--hostname", "0.0.0.0", "--port", "7860"], env)


if __name__ == "__main__":
    main()
