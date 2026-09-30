"""Deploy a digest-pinned image to an existing private Docker Space."""

import argparse
import json
import re
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, build_hf_headers, validate_repo_id

from s1mb.result_repository import ModelMetadata


def model_name(url: str) -> str | None:
    """Normalize a model page or checkpoint URL to its Hub repository ID."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != "huggingface.co":
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2 or parts[0] in {"datasets", "spaces", "collections", "blog", "docs"}:
        return None
    if len(parts) > 2 and (parts[2] not in {"tree", "resolve", "blob"} or len(parts) < 4):
        return None
    name = "/".join(parts[:2])
    validate_repo_id(name)
    return name


def published_models(api: HfApi, repo: str) -> list[str]:
    """Read only bounded display metadata at one resolved results revision."""
    info = api.dataset_info(repo, revision="main", files_metadata=True)
    if not info.sha or not re.fullmatch(r"[a-f0-9]{40}", info.sha):
        raise ValueError("Results dataset must resolve to an exact commit SHA")
    files = {file.rfilename: file for file in info.siblings}
    measured_folders = {
        PurePosixPath(name).parts[0] for name in files
        if len(PurePosixPath(name).parts) == 2 and name.endswith(".json.xz")
    }
    names = set()
    for folder in sorted(measured_folders):
        filename = f"{folder}/metadata.json"
        metadata_file = files.get(filename)
        if metadata_file is None:
            raise ValueError(f"Missing model metadata: {filename}")
        if metadata_file.size is None or not 0 < metadata_file.size <= 65536:
            raise ValueError(f"Invalid metadata size: {filename}")
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=info.sha,
                                    token=api.token))
        if path.stat().st_size != metadata_file.size:
            raise ValueError(f"Metadata size mismatch: {filename}")
        metadata = ModelMetadata.model_validate_json(path.read_bytes())
        if metadata.model_id != folder:
            raise ValueError(f"Model directory/metadata ID mismatch: {filename}")
        url = metadata.hf_url or metadata.url
        if url:
            name = model_name(url)
            if metadata.hf_url and name is None:
                raise ValueError(f"Invalid Hugging Face model link: {filename}")
            if name:
                names.add(name)
    print(f"Read model references from {repo}@{info.sha}", flush=True)
    return sorted(names, key=lambda name: (name.casefold(), name))


def render_models(names: list[str]) -> bytes:
    """Generate a stable, inert Python list for Hub model discovery."""
    for name in names:
        validate_repo_id(name)
        if name.count("/") != 1:
            raise ValueError("Model references require owner/model IDs")
    ordered = sorted(set(names), key=lambda name: (name.casefold(), name))
    header = '''"""Auto-generated Hugging Face model references for the S1MB leaderboard.

The Space displays saved results and does not import this file or run inference.
Literal model IDs let the Hub associate this Space with model repositories.
Generated from published results metadata; do not edit by hand.
"""

MODEL_NAMES = [
'''
    return (header + "".join(f"    {json.dumps(name)},\n" for name in ordered) + "]\n").encode()


def payload(image: str, source_sha: str, models: list[str]) -> dict[str, bytes]:
    if not re.fullmatch(r"ghcr\.io/[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}", image):
        raise ValueError("Use a GHCR image pinned by SHA-256 digest")
    if not re.fullmatch(r"[a-f0-9]{40}", source_sha):
        raise ValueError("Use the full source commit SHA")
    return {
        "README.md": Path(__file__).with_name("README.md").read_bytes(),
        "Dockerfile": f"# Source commit: {source_sha}\nFROM {image}\n".encode(),
        "models.py": render_models(models),
    }


def check_space(api: HfApi, space: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/S1MB-leaderboard", space):
        raise ValueError("Configure OWNER/S1MB-leaderboard as the Space repository")
    info = api.space_info(space)
    if info.private is not True or info.sdk != "docker":
        raise ValueError("Deployment requires an existing private Docker Space")
    volumes = info.runtime.raw.get("volumes", []) if info.runtime else []
    if not any(v.get("type") == "dataset" and v.get("source") == "hotchpotch/s1mb-result"
               and v.get("mountPath") == "/mnt/results" and v.get("readOnly") is True
               for v in volumes):
        raise ValueError("Mount the read-only results Dataset at /mnt/results before deploying")
    if not any(v.get("type") == "bucket" and v.get("mountPath") == "/mnt/cache"
               and v.get("readOnly") is not True for v in volumes):
        raise ValueError("Mount a writable cache Bucket at /mnt/cache before deploying")
    return info


def deploy(api: HfApi, space: str, files: dict[str, bytes], source_sha: str) -> str:
    info = check_space(api, space)
    operations = []
    for name, data in files.items():
        try:
            current = Path(hf_hub_download(space, name, repo_type="space", revision=info.sha,
                                           token=api.token)).read_bytes()
        except EntryNotFoundError:
            current = None
        if current != data:
            operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=data))
    if not operations:
        print("Space deployment files are unchanged", flush=True)
        return info.sha
    commit = api.create_commit(
        repo_id=space, repo_type="space", revision="main", parent_commit=info.sha,
        operations=operations,
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
    parser.add_argument("--output", type=Path, help="Render files locally without modifying the Space")
    parser.add_argument("--results-repo", default="hotchpotch/s1mb-result")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if bool(args.output) == args.deploy:
        parser.error("Choose exactly one of --output or --deploy")
    api = HfApi()
    files = payload(args.image, args.source_sha, published_models(api, args.results_repo))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (args.output / name).write_bytes(data)
        return
    revision = deploy(api, args.space, files, args.source_sha)
    print(f"Deployment target Space commit: {revision}", flush=True)
    wait_ready(api, args.space, revision, args.timeout)


if __name__ == "__main__":
    main()
