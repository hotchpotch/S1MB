"""Meta Encoder text-choice embeddings and calibrated candidate probabilities."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import math
import os
import platform
from collections import OrderedDict
from typing import Any

from s1mb.data import InferenceCase, ModelInfo, Option, Prediction, Question, check_probabilities
from s1mb.parameters import parameter_metadata

from .upstream import checkpoint_path

PREFIX = "Select the correct option."
SYSTEM_PROMPT = "Represent the user's input."
DEFAULT_CONTEXT_LIMIT = 8192
DEFAULT_CANDIDATE_BATCH_SIZE = 8
DEFAULT_CACHE_CAPACITY = 4096


def _parsed_text(raw: str | None, fallback: str) -> str:
    """Match the released text-choice conversion for JSON-backed fields."""
    return str(json.loads(raw)).strip() if raw is not None else fallback.strip()


def _state_text(case: InferenceCase) -> str:
    if case.state_json is not None:
        return case.state_json.strip()
    return json.dumps(case.state, ensure_ascii=False, separators=(",", ":"))


def render_query(case: InferenceCase, question: Question) -> str:
    """Render one S1MB decision using Meta Encoder's released text-choice protocol."""
    criteria = "\n".join(
        f"- {option.id}: {_parsed_text(option.description_json, option.description)}"
        for option in question.options
    )
    labels = ", ".join(option.id for option in question.options)
    body = "\n".join(
        [
            _state_text(case),
            "",
            f"Task: {_parsed_text(question.instructions_json, question.instructions)}",
            "Criteria:",
            criteria,
            f"Options: {labels}",
        ]
    ).strip()
    return f"{PREFIX} {body}"


def render_candidate(option: Option) -> str:
    """Use the declared option ID verbatim as the independently encoded candidate."""
    return option.id


def chat_messages(text: str) -> list[dict[str, Any]]:
    """Build the model's text-only chat after its reserved-token sanitation."""
    text = text.replace("<|patch|>", " ").replace("<|video|>", " ")
    text = " ".join(text.split())
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


class MetaEncoderAdapter:
    """Embed each query and declared candidate, then normalize cosine scores."""

    case_batch_size = 1
    candidate_batch_size = DEFAULT_CANDIDATE_BATCH_SIZE
    cache_capacity = DEFAULT_CACHE_CAPACITY

    def __init__(
        self,
        model: str,
        revision: str,
        device: str,
        *,
        temperature: float,
        context_limit: int | None = None,
        attention: str = "sdpa",
    ) -> None:
        if not device.startswith("cuda"):
            raise ValueError("Meta Encoder requires explicit CUDA; refusing CPU fallback")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Meta Encoder temperature must be finite and positive")
        if context_limit is not None and context_limit < 1:
            raise ValueError("Meta Encoder context limit must be positive")
        if attention not in {"sdpa", "flash_attention_2"}:
            raise ValueError("Meta Encoder attention must be SDPA or FlashAttention 2")

        self.model_id = model
        self.device = device
        self.temperature = temperature
        self.context_limit = context_limit or DEFAULT_CONTEXT_LIMIT
        self.attention = attention
        self.path, self.revision = checkpoint_path(model, revision)
        self.torch = importlib.import_module("torch")
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
        self.torch.cuda.set_device(device)

        transformers = importlib.import_module("transformers")
        modeling = importlib.import_module(
            "transformers.models.muse_glimmer.modeling_muse_glimmer"
        )
        model_class = modeling.MuseGlimmerForConditionalGeneration
        self.processor = transformers.AutoProcessor.from_pretrained(str(self.path))
        if not getattr(self.processor, "tokenizer", None):
            raise ValueError("Meta Encoder checkpoint did not provide a tokenizer")
        self.processor.tokenizer.padding_side = "left"
        self.engine = model_class.from_pretrained(
            str(self.path),
            dtype=self.torch.bfloat16,
            device_map="auto",
            attn_implementation=attention,
            low_cpu_mem_usage=True,
        )
        self.engine.config.use_cache = False
        self.engine.eval()
        text_config = getattr(self.engine.config, "text_config", self.engine.config)
        self.checkpoint_context_limit = getattr(text_config, "max_position_embeddings", None)
        if (
            self.checkpoint_context_limit is not None
            and self.context_limit > self.checkpoint_context_limit
        ):
            raise ValueError("Meta Encoder context limit exceeds the checkpoint capacity")
        for module in self.engine.modules():
            for name, buffer in module.named_buffers(recurse=False):
                if "inv_freq" in name and buffer.dtype != self.torch.bfloat16:
                    setattr(module, name, buffer.to(self.torch.bfloat16))

        self._parameter_metadata = parameter_metadata(self.engine)
        self._candidate_cache: OrderedDict[str, Any] = OrderedDict()

    def _chat(self, text: str) -> str:
        return self.processor.apply_chat_template(
            chat_messages(text),
            add_generation_prompt=False,
            tokenize=False,
        )

    def _encode(self, texts: list[str], batch_size: int) -> Any:
        outputs = []
        for start in range(0, len(texts), batch_size):
            chats = [self._chat(text) for text in texts[start : start + batch_size]]
            inputs = self.processor(
                text=chats,
                images=None,
                videos=None,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            if inputs["input_ids"].shape[1] > self.context_limit:
                raise ValueError("Meta Encoder input exceeds context limit; refusing truncation")
            inputs["input_ids"] = inputs["input_ids"].long()
            inputs["attention_mask"] = inputs["attention_mask"].long()
            inputs = {key: value.to(self.engine.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                output = self.engine(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
                pooled = output.hidden_states[-1][:, -1, :]
                outputs.append(self.torch.nn.functional.normalize(pooled.float(), p=2, dim=-1))
        return self.torch.cat(outputs)

    def _candidate_embeddings(self, texts: list[str]) -> Any:
        missing = list(dict.fromkeys(text for text in texts if text not in self._candidate_cache))
        if missing:
            embeddings = self._encode(missing, min(len(missing), self.candidate_batch_size))
            for text, embedding in zip(missing, embeddings, strict=True):
                self._candidate_cache[text] = embedding
                self._candidate_cache.move_to_end(text)
                if len(self._candidate_cache) > self.cache_capacity:
                    self._candidate_cache.popitem(last=False)
        for text in texts:
            self._candidate_cache.move_to_end(text)
        return self.torch.stack([self._candidate_cache[text] for text in texts])

    def _render_query(self, case: InferenceCase, question: Question) -> str:
        return render_query(case, question)

    def predict(self, case: InferenceCase) -> list[Prediction]:
        predictions = []
        for question in case.questions:
            candidates = [render_candidate(option) for option in question.options]
            candidate_embeddings = self._candidate_embeddings(candidates)
            query_embedding = self._encode([self._render_query(case, question)], 1)
            cosine = (query_embedding @ candidate_embeddings.T)[0]
            values = self.torch.softmax(cosine / self.temperature, dim=-1).cpu().tolist()
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=question.id,
                    probabilities=probabilities,
                )
            )
        return predictions

    def metadata(self) -> ModelInfo:
        versions = {"python": platform.python_version()}
        packages = {
            "accelerate": "accelerate",
            "flash_attn": "flash-attn",
            "huggingface_hub": "huggingface-hub",
            "pillow": "pillow",
            "safetensors": "safetensors",
            "tokenizers": "tokenizers",
            "torch": "torch",
            "torchvision": "torchvision",
            "transformers": "transformers",
        }
        for key, package in packages.items():
            try:
                versions[key] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        return ModelInfo(
            **self._parameter_metadata,
            id=self.model_id,
            adapter="meta-encoder",
            revision=self.revision,
            settings={
                "device": self.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "dtype": "bfloat16",
                "model_class": type(self.engine).__name__,
                "model_config_class": type(self.engine.config).__name__,
                "processor_class": type(self.processor).__name__,
                "requested_attention_implementation": self.attention,
                "attention_implementation": getattr(
                    self.engine.config, "_attn_implementation", self.attention
                ),
                "device_map": "auto",
                "max_input_tokens": self.context_limit,
                "checkpoint_context_limit": self.checkpoint_context_limit,
                "input_length_policy": "reject-overflow",
                "padding_side": self.processor.tokenizer.padding_side,
                "truncation_side": self.processor.tokenizer.truncation_side,
                "case_batch_size": self.case_batch_size,
                "task_batch_size": 1,
                "candidate_batch_size": self.candidate_batch_size,
                "candidate_cache_capacity": self.cache_capacity,
                "renderer": "text-choice-option-id-v1",
                "query_prefix": PREFIX,
                "chat_system_prompt": SYSTEM_PROMPT,
                "state_serialization": "raw-state-json",
                "structured_text_serialization": "python-str-of-json",
                "reserved_media_tokens": "replace-with-space",
                "whitespace_policy": "collapse",
                "pooling": "last-hidden-state-final-token",
                "embedding_output": "float32-l2-normalized",
                "embedding_output_dtype": "float32",
                "normalization": "l2",
                "probability_rule": "joint-cosine-softmax",
                "temperature": self.temperature,
                "runtime_versions": versions,
            },
        )

    def close(self) -> None:
        self._candidate_cache.clear()
        if hasattr(self, "engine"):
            del self.engine
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        self.torch.cuda.empty_cache()
