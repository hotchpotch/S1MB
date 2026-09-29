"""Evaluate an explicitly selected Laya checkpoint, without automatic routing."""

import importlib
import importlib.metadata
from pathlib import Path

from s1mb.data import InferenceCase, ModelInfo, Prediction
from s1mb.parameters import parameter_metadata

from .base import questions_for_api
from .upstream import candidate_batches


def full_sequence(tokenizer, state, question, common, max_len, head_max_len=None):
    """Keep the native token layout without shortening any field."""
    mask = tokenizer.mask_token
    texts = [
        f"{question['t']} question: {question['ins']}".replace(mask, " "),
        *[" " + text.replace(mask, " ") for text in common.render_options(question)],
        common.serialize_state(state).replace(mask, " "),
    ]
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    ids = [tokenizer.cls_token_id, *encoded[0], tokenizer.sep_token_id]
    markers = []
    for option in encoded[1:-1]:
        markers.append(len(ids))
        ids.extend([tokenizer.mask_token_id, *option])
    if head_max_len is not None and len(ids) - 2 > head_max_len:
        raise ValueError("Laya question and candidates exceed head_max_len; refusing truncation")
    ids.extend([tokenizer.sep_token_id, *encoded[-1], tokenizer.sep_token_id])
    if len(ids) > max_len:
        raise ValueError(f"Laya input requires {len(ids)} tokens, exceeding {max_len}")
    return {"ids": ids, "markers": markers, "qtype": common.QTYPES[question["t"]]}


class LayaAdapter:
    case_batch_size = 32

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
        if device.startswith("cuda") and self.agent.device.type != "cuda":
            raise RuntimeError("Laya failed to load on CUDA; refusing CPU inference")
        self.common = importlib.import_module("laya.common")
        self.collate = importlib.import_module("laya.agent").collate_items
        self.initial_device = str(self.agent.device)
        self.torch = torch
        # Keep the checkpoint's FP32 weights and arithmetic. BF16 autocast can
        # amplify batch-shape differences through the calibrated decision head.
        self.agent.dtype = torch.float32
        self.agent.amp_enabled = False
        torch.backends.mha.set_fastpath_enabled(False)
        self.max_len = max_len or self.agent.cfg.get("max_len", 512)
        self.head_max_len = head_max_len or self.agent.cfg.get("head_max_len", 192)
        if not 0 < self.head_max_len < self.max_len:
            raise ValueError("Require 0 < head_max_len < max_len")

    def predict(self, case: InferenceCase) -> list[Prediction]:
        return self.predict_batch([case])[0]

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        rows = []
        for index, case in enumerate(cases):
            for q in case.questions:
                native = questions_for_api([q], structured=True)[q.id]
                # Stable dataset identifiers must not become model text.
                if q.task == "choice":
                    native["criteria"] = {
                        f"option_{i}": value for i, value in enumerate(native["criteria"].values())
                    }
                internal = self.agent._to_internal(native)
                item = full_sequence(
                    self.agent.tok,
                    case.state,
                    internal,
                    self.common,
                    self.max_len,
                    self.head_max_len,
                )
                rows.append((index, case.case_id, q, item))
        rows.sort(key=lambda row: len(row[3]["ids"]))
        results: list[dict[str, Prediction]] = [{} for _ in cases]
        with (
            self.torch.inference_mode(),
            self.torch.autocast(
                self.agent.device.type, dtype=self.agent.dtype, enabled=self.agent.amp_enabled
            ),
        ):
            for group in candidate_batches(
                rows,
                [len(row[3]["ids"]) for row in rows],
                max_batch=self.questions_per_call or 32,
                token_budget=8192,
            ):
                batch = self.collate([[row[3]] for row in group], self.agent.tok.pad_token_id)
                # Direct model invocation cannot silently fall back to CPU on OOM.
                logits, _ = self.agent.model(
                    *[
                        batch[k].to(self.agent.device)
                        for k in (
                            "input_ids",
                            "attention_mask",
                            "marker_pos",
                            "marker_mask",
                            "qtype",
                        )
                    ]
                )
                values = []
                for row_index, (_, _, q, item) in enumerate(group):
                    count = len(item["markers"])
                    bucket = self.common.temp_bucket(item["qtype"], count)
                    temperature = self.agent.temperature_by_options.get(
                        bucket, self.agent.temperature[item["qtype"]]
                    )
                    values.append((logits[row_index, :count].float() / temperature).softmax(-1))
                probabilities = self.torch.cat(values).cpu().split([len(v) for v in values])
                for (index, case_id, q, _), p in zip(group, probabilities, strict=True):
                    ids = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
                    results[index][q.id] = Prediction(
                        case_id=case_id,
                        question_id=q.id,
                        probabilities=dict(zip(ids, p.tolist(), strict=True)),
                    )
        return [
            [result[q.id] for q in case.questions]
            for result, case in zip(results, cases, strict=True)
        ]

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            **parameter_metadata(self.agent),
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
                "renderer": "native-layout-full-fields-anonymous-choice-v2",
                "input_length_policy": "reject-overflow",
                "case_batch_size": self.case_batch_size,
                "microbatch_tokens": 8192,
                "max_len": self.max_len,
                "head_max_len": self.head_max_len,
                "probabilities": "native calibrated FP32 softmax without API rounding",
            },
        )

    def close(self) -> None:
        del self.agent
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
