"""Span-02's native Rust/CUDA server with its deterministic released policy."""

import base64
import hashlib
import io
import ipaddress
import os
import subprocess
import tempfile
import time
from pathlib import Path

from s1mb.data import ModelInfo

from .base import decode_answers
from .upstream import checkpoint_path
from .vision_native import ordered_images, typed_questions
from .winnow import WinnowAdapter, check_port_available


class SpanAdapter(WinnowAdapter):
    def __init__(self, model, revision, source, device, base_revision,
                 server_host="127.0.0.1", server_port=8100, context_limit=32768):
        address = ipaddress.ip_address(server_host)
        if address.version != 4 or not (address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")):
            raise ValueError("Span must bind to localhost or a Tailscale IPv4 address")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        if device != "cuda:0" or gpu not in {"0", "1"}:
            raise ValueError("Span requires one explicitly selected CUDA device")
        if context_limit < 1:
            raise ValueError("Span context limit must be positive")
        root = Path(source).resolve(strict=True)
        binary = root / "target/release/dllm-server"
        path, resolved = checkpoint_path(model, revision)
        draft_model = "google/gemma-4-26B-A4B-it-assistant"
        draft, draft_revision = checkpoint_path(draft_model, base_revision)
        with binary.open("rb") as stream:
            binary_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        native_sha = hashlib.sha256()
        for name in sorted(root.glob("crates/**/src/*.rs")) + sorted(root.glob("crates/kernels/cuda/*")):
            native_sha.update(str(name.relative_to(root)).encode())
            native_sha.update(name.read_bytes())
        self.info = ModelInfo(id=model, adapter="span", revision=resolved, settings={
            "device": device, "cuda_visible_devices": gpu,
            "server_binary_sha256": binary_sha, "source_native_sha256": native_sha.hexdigest(),
            "draft_model": draft_model, "draft_revision": draft_revision,
            "dtype": "native-NVFP4-FP8", "max_input_tokens": context_limit,
            "input_length_policy": "reject-overflow-native", "max_images": 4,
            "reasoning": "released-auto-deterministic-750ms-cost-model-cap1024",
            "renderer": "native-span-typed-anonymous-choice-numeric-score-v1",
            "image_soft_tokens": 560, "kv_budget_gb": 9,
            "media": "native-assets-base64-v1",
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "parameter_count_condition": "quantized-engine-count-unavailable",
        })
        self.image_receipts = {}
        check_port_available(server_host, server_port)
        self.url = f"http://{server_host}:{server_port}"
        self.log = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 -- closed with server
        self.process = subprocess.Popen([
            str(binary), "--bind", f"{server_host}:{server_port}", "--model", str(path),
            "--spec-draft", str(draft), "--max-len", str(context_limit), "--budget-ms", "0",
            "--tau", "0.8", "--cap", "1024", "--think-ms", "750",
            "--think-cost", str(path / "think_cost.json"), "--image-soft-tokens", "560",
            "--serial", "--no-prefix-cache", "--spec-max-branches", "16", "--kv-gb", "9",
        ], stdout=self.log, stderr=subprocess.STDOUT)
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
                raise RuntimeError("Span server did not become healthy within 180 seconds")
        except BaseException:
            self.close()
            raise

    def predict(self, case):
        if len(case.images) > 4:
            raise ValueError("Span accepts at most four images; refusing image removal")
        images, state, receipts = ordered_images(case)
        assets = {}
        for index, image in enumerate(images, 1):
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            assets[f"<image:{index}>"] = {"type": "image", "format": "png",
                                         "data_base64": base64.b64encode(stream.getvalue()).decode("ascii")}
        if images:
            self.image_receipts[case.case_id] = receipts
        answers = {}
        for key, definition in typed_questions(case).items():
            result = self.request("/v1/systemone", {
                "state": state if images else case.state, "questions": {key: definition},
                **({"assets": assets} if assets else {}),
            })
            if set(result["answers"]) != {key}:
                raise ValueError("Span returned incorrect question alignment")
            answers[key] = result["answers"][key]
        return decode_answers(case, answers, anonymous_choice=True)
