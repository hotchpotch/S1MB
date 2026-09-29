"""Evaluate Hugging Face Bekko models through their remote typed prediction API."""

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

from s1mb.data import InferenceCase, ModelInfo, Prediction, check_probabilities
from s1mb.parameters import parameter_metadata


def native_input(case: InferenceCase) -> dict:
    """Reconstruct only the structured inference fields for the remote typed prediction API."""
    return {
        "state_json": json.dumps(
            case.state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "decisions": [
            {
                "id": q.id,
                "kind": "judgment",
                "type": q.task,
                "instructions_json": q.instructions_json or json.dumps(q.instructions),
                "system_prompt": q.system_prompt,
                "criteria": [
                    {
                        "id": o.id,
                        "description_json": o.description_json or json.dumps(o.description),
                        "value": o.value,
                    }
                    for o in q.options
                ],
                "documents": [],
                "scoring": None,
            }
            for q in case.questions
        ],
    }


def resolve_checkpoint(checkpoint: str, revision: str | None):
    """Resolve Hub references once so code, weights and provenance share one SHA."""
    path = Path(checkpoint)
    if path.exists():
        raise ValueError("Bekko evaluation requires a Hugging Face model ID, not a local path")
    from huggingface_hub import HfApi, snapshot_download

    sha = HfApi().model_info(checkpoint, revision=revision).sha
    if not sha:
        raise ValueError("Hub did not return a model commit SHA")
    snapshot = Path(snapshot_download(checkpoint, revision=sha))
    return snapshot, checkpoint, sha


class BekkoV0Adapter:
    def __init__(
        self,
        checkpoint: str,
        device: str,
        query_length: int | None = None,
        document_length: int | None = None,
        *,
        revision: str | None = None,
        token_budget: int = 64_000,
        context_length: int | None = None,
        case_batch_size: int = 128,
        compile_model: bool = False,
        cpu_smoke: bool = False,
    ):
        if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("Bekko v0 requires an explicit CPU smoke or CUDA device")
        if device == "cpu" and not cpu_smoke:
            raise ValueError("CPU inference is restricted to explicit smoke checks")
        if token_budget < 1 or case_batch_size < 1:
            raise ValueError("Batch size and token budget must be positive")
        self.checkpoint, self.model_id, self.hub_revision = resolve_checkpoint(checkpoint, revision)
        self.torch = importlib.import_module("torch")
        dynamic = importlib.import_module("transformers.dynamic_module_utils")
        source = self.model_id
        model_class = dynamic.get_class_from_dynamic_module(
            "inference_v0.BekkoSentenceTransformer",
            source,
            revision=self.hub_revision,
            code_revision=self.hub_revision,
            local_files_only=True,
        )
        self.model = model_class(
            source,
            revision=self.hub_revision,
            device=device,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        self.runtime = self.model[0]
        if getattr(self.runtime, "budget_policy", None) != "adaptive-v1":
            raise ValueError("Re-export Bekko v0 with adaptive-v1 input budgeting")
        if not {"choice", "noul", "score"} <= set(self.runtime.tasks):
            raise ValueError("Bekko checkpoint must have Choice, Noul and Score heads")
        self.context_length = (
            self.runtime.context_length if context_length is None else context_length
        )
        self.query_length = (
            min(self.runtime.query_length, self.context_length - 2)
            if query_length is None
            else query_length
        )
        self.document_length = (
            min(self.runtime.document_length, self.context_length - 3)
            if document_length is None
            else document_length
        )
        # Native empty prediction validates positional capacity without running a forward pass.
        self.model.predict(
            [],
            query_length=self.query_length,
            document_length=self.document_length,
            context_length=self.context_length,
            show_progress_bar=False,
        )
        self.device = device
        self.token_budget = token_budget
        self.case_batch_size = case_batch_size
        self.compile_model = compile_model
        self.cpu_smoke = cpu_smoke
        if compile_model:
            self.model.compile_inference()
        self.file_hashes = {}
        for file in sorted(self.checkpoint.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                with file.open("rb") as stream:
                    self.file_hashes[str(file.relative_to(self.checkpoint))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        self.content_hash = hashlib.sha256(
            json.dumps(self.file_hashes, sort_keys=True).encode()
        ).hexdigest()
        self.revision = self.hub_revision
        self.parameters = parameter_metadata(self.model)

    def predict(self, case: InferenceCase) -> list[Prediction]:
        return self.predict_batch([case])[0]

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        outputs: list[list[Prediction]] = [[] for _ in cases]
        for start in range(0, len(cases), self.case_batch_size):
            chunk = cases[start : start + self.case_batch_size]
            results = self.model.predict(
                [native_input(case) for case in chunk],
                batch_size=self.case_batch_size,
                token_budget=self.token_budget,
                query_length=self.query_length,
                document_length=self.document_length,
                context_length=self.context_length,
                show_progress_bar=False,
            )
            for index, (case, result) in enumerate(zip(chunk, results, strict=True), start):
                if set(result) != {question.id for question in case.questions}:
                    raise ValueError("Native output has missing or unexpected decision IDs")
                for question in case.questions:
                    ids = [o.id for o in question.options]
                    probabilities = result[question.id]["probabilities"]
                    check_probabilities(probabilities, ids)
                    outputs[index].append(
                        Prediction(
                            case_id=case.case_id,
                            question_id=question.id,
                            probabilities={key: probabilities[key] for key in ids},
                        )
                    )
        return outputs

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            **self.parameters,
            id=self.model_id,
            adapter="bekko-v0",
            revision=self.revision,
            settings={
                "checkpoint_files_sha256": self.file_hashes,
                "checkpoint_content_sha256": self.content_hash,
                "loader": "inference_v0.BekkoSentenceTransformer",
                "prediction_api": "model.predict(native_inputs)",
                "hub_repo_id": self.model_id,
                "hub_revision": self.hub_revision,
                "checkpoint_encoder_config": self.runtime.settings,
                "query_length": self.query_length,
                "document_length": self.document_length,
                "input_length_policy": "native-adaptive-v1-truncate",
                "max_position_embeddings": self.runtime.encoder.backbone.config.max_position_embeddings,
                "query_truncation": self.runtime.query_truncation,
                "candidate_truncation": "right",
                "budget_allocation": "half-shares-lend-unused-candidate-odd-token",
                "context_length": self.context_length,
                "device": self.device,
                "cpu_smoke": self.cpu_smoke,
                "inference": "inference_v0",
                "attention": "sdpa",
                "dtype": "float32" if self.device == "cpu" else "bfloat16-autocast",
                "compile": self.compile_model,
                "case_batch_size": self.case_batch_size,
                "microbatch_tokens": self.token_budget,
                "prefix_layout": self.runtime.prefix_layout,
                "torch_threads": self.torch.get_num_threads(),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "sentence-transformers",
                        "safetensors",
                    )
                },
            },
        )

    def close(self) -> None:
        del self.runtime
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
