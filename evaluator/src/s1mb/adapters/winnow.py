"""Pinned Winnow GGUF inference through its independently built CUDA server."""

import hashlib
import importlib
import ipaddress
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from s1mb.data import ModelInfo

from .base import decode_answers, questions_for_api
from .upstream import source_path


def check_port_available(host, port):
    """Reject active listeners while allowing a just-closed server's TIME_WAIT sockets."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))


class WinnowAdapter:
    case_batch_size = 1

    def __init__(
        self,
        model,
        revision,
        source,
        device,
        server_host="127.0.0.1",
        server_port=8091,
        max_candidates=None,
    ):
        capacity = 64 if max_candidates is None else max_candidates
        if not 2 <= capacity <= 255:
            raise ValueError("Winnow candidate capacity must be within 2..255")
        address = ipaddress.ip_address(server_host)
        if address.version != 4 or not (
            address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")
        ):
            raise ValueError("Winnow must bind to localhost or a Tailscale IPv4 address")
        if not device.startswith("cuda"):
            raise ValueError("Winnow requires explicit CUDA; no CPU fallback")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not gpu or "," in gpu or device not in {"cuda", "cuda:0"}:
            raise ValueError("Winnow requires exactly one explicitly visible GPU")
        root, digest = source_path(source)
        native_digest = hashlib.sha256()
        for file in sorted((root / "native").rglob("*")):
            if file.is_file():
                native_digest.update(str(file.relative_to(root)).encode())
                native_digest.update(file.read_bytes())
        with (root / ".build/bin/winnow-server").open("rb") as binary:
            binary_digest = hashlib.file_digest(binary, "sha256").hexdigest()
        manifest = json.loads((root / "manifests/models.json").read_text())
        artifact = manifest["release"]["model"]
        hub = importlib.import_module("huggingface_hub")
        resolved = hub.model_info(model, revision=revision).sha
        if not resolved:
            raise ValueError("Missing resolved Winnow revision")
        path = Path(hub.hf_hub_download(model, artifact["file"], revision=resolved))
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
        if sha.hexdigest() != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError("Winnow GGUF differs from the runtime's verified release manifest")
        self.info = ModelInfo(
            id=model,
            adapter="winnow",
            revision=resolved,
            settings={
                "device": device,
                "cuda_visible_devices": gpu,
                "source_python_sha256": digest,
                "source_native_sha256": native_digest.hexdigest(),
                "server_binary_sha256": binary_digest,
                "runtime_lock": json.loads((root / "runtime.lock.json").read_text()),
                "weights_sha256": artifact["sha256"],
                "dtype": "Q8_0-weights-Q8_0-KV",
                "attention_implementation": "llama.cpp-flash-attention",
                "input_length_policy": "reject-overflow",
                "max_input_tokens": 65536,
                "max_candidates": capacity,
                "checkpoint_max_candidates": 64,
                "case_batch_size": 1,
                "questions_per_call": 1,
                "renderer": "native-systemone-structured-anonymous-choice-v1",
                "temperature": 1.0,
                "head": "selected",
                "prefix_reuse": True,
            },
        )
        check_port_available(server_host, server_port)
        self.url = f"http://{server_host}:{server_port}"
        self.log = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 -- closed with the server
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(root / "scripts/serve.py"),
                "--model",
                str(path),
                "--text-only",
                "--gpu",
                gpu,
                "--host",
                server_host,
                "--port",
                str(server_port),
                "--context",
                "65536",
                "--decision-parallel",
                "1",
                "--chat-parallel",
                "1",
                "--cache",
                "q8_0",
                "--memory",
                "exclusive",
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    raise RuntimeError(self.log.read().decode(errors="replace")[-8000:])
                try:
                    self.request("/health")
                    break
                except (OSError, ValueError):
                    time.sleep(0.5)
            else:
                raise RuntimeError("Winnow server did not become healthy within 180 seconds")
        except BaseException:
            self.close()
            raise

    def request(self, path, body=None):
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300 if body is not None else 2) as response:
            return json.load(response)

    def metadata(self):
        return self.info

    def predict(self, case):
        answers = {}
        for key, question in questions_for_api(
            case.questions, structured=True, anonymous_choice=True
        ).items():
            if len(question["criteria"]) > self.info.settings["max_candidates"]:
                raise ValueError("Winnow candidate capacity exceeded; refusing truncation")
            result = self.request(
                "/v1/systemone",
                {"state": case.state, "questions": {"decision": question}},
            )
            answers[key] = result["answers"]["decision"]
        return decode_answers(case, answers, anonymous_choice=True)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()
