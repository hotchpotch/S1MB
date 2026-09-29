"""Deploy a digest-pinned image to an existing private Docker Space."""

import argparse
import re
import time
from pathlib import Path

import httpx
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import build_hf_headers


def payload(image: str, source_sha: str) -> dict[str, bytes]:
    if not re.fullmatch(r"ghcr\.io/[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}", image):
        raise ValueError("Use a GHCR image pinned by SHA-256 digest")
    if not re.fullmatch(r"[a-f0-9]{40}", source_sha):
        raise ValueError("Use the full source commit SHA")
    return {
        "README.md": Path(__file__).with_name("README.md").read_bytes(),
        "Dockerfile": f"# Source commit: {source_sha}\nFROM {image}\n".encode(),
    }


def check_space(api: HfApi, space: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/S1MB-leaderboard", space):
        raise ValueError("Configure OWNER/S1MB-leaderboard as the Space repository")
    info = api.space_info(space)
    if info.private is not True or info.sdk != "docker":
        raise ValueError("Deployment requires an existing private Docker Space")
    variables = api.get_space_variables(space)
    if not variables.get("S1MB_HF_RESULTS_REPO"):
        raise ValueError("Configure the Space results Dataset before deploying")
    return info


def deploy(api: HfApi, space: str, files: dict[str, bytes], source_sha: str) -> str:
    info = check_space(api, space)
    commit = api.create_commit(
        repo_id=space, repo_type="space", revision="main", parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
                    for name, data in files.items()],
        commit_message=f"Deploy S1MB viewer {source_sha}",
    )
    return commit.oid


def wait_ready(api: HfApi, space: str, revision: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = api.space_info(space)
        if info.private is not True:
            raise RuntimeError("Space is no longer private")
        if info.sha != revision:
            raise RuntimeError("Space was updated by another deployment")
        runtime = api.get_space_runtime(space)
        stage = str(runtime.stage)
        print(f"Space stage: {stage}", flush=True)
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "PAUSED"}:
            raise RuntimeError(f"Space deployment failed: {stage}")
        # RUNNING alone can still refer to the preceding image during a rebuild.
        if stage == "RUNNING" and runtime.raw.get("sha") == revision:
            host = getattr(info, "host", None)
            if not host or not re.fullmatch(r"https://[a-z0-9-]+\.hf\.space", host):
                raise RuntimeError("Space did not return a supported application URL")
            try:
                response = httpx.get(host + "/", headers=build_hf_headers(token=api.token),
                                     timeout=60, follow_redirects=False)
            except httpx.TransportError:
                time.sleep(15)
                continue
            if response.status_code == 200 and "S1MB" in response.text:
                print(f"Verified private Space revision {revision}", flush=True)
                return
        time.sleep(15)
    raise TimeoutError("Space did not become ready before the deployment deadline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, help="Render files locally without calling the Hub")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    files = payload(args.image, args.source_sha)
    if bool(args.output) == args.deploy:
        parser.error("Choose exactly one of --output or --deploy")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (args.output / name).write_bytes(data)
        return
    api = HfApi()
    revision = deploy(api, args.space, files, args.source_sha)
    print(f"Updated Space commit: {revision}", flush=True)
    wait_ready(api, args.space, revision, args.timeout)


if __name__ == "__main__":
    main()
