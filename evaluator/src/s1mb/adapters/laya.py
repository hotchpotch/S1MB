"""Evaluate an explicitly selected Laya checkpoint, without automatic routing."""

import importlib
import importlib.metadata
from pathlib import Path

from s1mb.data import InferenceCase, ModelInfo, Prediction

from .base import decode_answers, questions_for_api


class LayaAdapter:
    def __init__(
        self,
        model: str,
        revision: str = "main",
        device: str = "cpu",
        max_len: int | None = None,
        head_max_len: int | None = None,
        questions_per_call: int | None = None,
        autocast_cache: bool = True,
    ):
        if questions_per_call is not None and questions_per_call < 1:
            raise ValueError("questions_per_call must be positive")
        self.questions_per_call = questions_per_call
        hub = importlib.import_module("huggingface_hub")
        laya = importlib.import_module("laya")
        torch = importlib.import_module("torch")
        torch.set_num_threads(4)
        torch.set_autocast_cache_enabled(autocast_cache)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        path = hub.snapshot_download(
            model,
            revision=revision,
            allow_patterns=[
                "rl_agent_config.json",
                "model.safetensors",
                "tokenizer/*",
                "encoder/*",
            ],
        )
        self.model_id, self.revision = model, Path(path).name
        self.agent = laya.Agent(path, device=device)
        self.initial_device = str(self.agent.device)
        self.torch = torch
        self.max_len = max_len or self.agent.cfg.get("max_len", 512)
        self.head_max_len = head_max_len or self.agent.cfg.get("head_max_len", 192)
        if not 0 < self.head_max_len < self.max_len:
            raise ValueError("Require 0 < head_max_len < max_len")

    def predict(self, case: InferenceCase) -> list[Prediction]:
        if str(self.agent.device) != self.initial_device:
            raise RuntimeError("Laya changed device during inference")
        answers = {}
        chunk = self.questions_per_call or len(case.questions)
        for start in range(0, len(case.questions), chunk):
            response = self.agent.system_one(
                case.state,
                questions_for_api(case.questions[start : start + chunk]),
                max_len=self.max_len,
                head_max_len=self.head_max_len,
            )
            if str(self.agent.device) != self.initial_device:
                raise RuntimeError(
                    "Laya changed device during inference; rerun explicitly on that device"
                )
            answers.update(response["answers"])
        return decode_answers(case, answers, rounded=True)

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            id=self.model_id,
            adapter="laya",
            revision=self.revision,
            settings={
                "laya_version": importlib.metadata.version("laya"),
                "torch_version": self.torch.__version__,
                "transformers_version": importlib.metadata.version("transformers"),
                "device": str(self.agent.device),
                "dtype": str(self.agent.dtype),
                "autocast_cache_enabled": self.torch.is_autocast_cache_enabled(),
                "questions_per_call": self.questions_per_call,
                "checkpoint_config": self.agent.cfg,
                "temperature": self.agent.temperature,
                "temperature_by_options": self.agent.temperature_by_options,
                "renderer": "reviewed-system-and-instruction-v1",
                "max_len": self.max_len,
                "head_max_len": self.head_max_len,
                "probabilities": "public API rounded to 4 decimals, renormalized within rounding tolerance",
            },
        )

    def close(self) -> None:
        del self.agent
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
