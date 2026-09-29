"""Evaluate exported Bekko inference_v0 checkpoints with native SDPA batching."""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

from s1mb.data import InferenceCase, ModelInfo, Prediction, check_probabilities
from s1mb.parameters import parameter_metadata

from .bekko import native_input
from .input_lengths import check_balanced_input_lengths_batched, check_input_lengths


def load_runtime(checkpoint: Path):
    """Load the reviewed standalone checkpoint code, without the training package."""
    file = checkpoint / "inference_v0.py"
    if not file.is_file():
        raise ValueError("Expected an exported v0 checkpoint containing inference_v0.py")
    name = "s1mb_bekko_v0_" + hashlib.sha256(file.read_bytes()).hexdigest()
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Bekko runtime: {file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    module = sys.modules[name]
    if not all(hasattr(module, attr) for attr in ("BekkoSentenceTransformer", "input_groups")):
        raise ValueError("Outdated inference_v0.py; re-export with the current Bekko export_v0")
    return module


def validate_groups(runtime, groups, query_length: int, document_length: int):
    """Match native token accounting, including optional candidate task markers."""
    if runtime.query_truncation == "balanced":
        check_balanced_input_lengths_batched(
            runtime.tokenizer, [g.query_parts for g in groups], query_length
        )
    else:
        check_input_lengths(
            runtime.tokenizer, [g.query for g in groups], [], query_length, document_length
        )
    for has_marker in (False, True):
        documents = [
            d
            for g in groups
            if (g.task in runtime.task_token_ids) == has_marker
            for d in g.candidates
        ]
        # The common validator reserves SEP; v0 may additionally prepend a task token.
        check_input_lengths(
            runtime.tokenizer, [], documents, query_length, document_length - int(has_marker)
        )


class BekkoV0Adapter:
    def __init__(
        self,
        checkpoint: str,
        device: str,
        query_length: int | None = None,
        document_length: int | None = None,
        *,
        token_budget: int = 64_000,
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
        self.checkpoint = Path(checkpoint).resolve()
        self.native = load_runtime(self.checkpoint)
        self.torch = importlib.import_module("torch")
        self.model = self.native.BekkoSentenceTransformer(
            str(self.checkpoint), device=device, trust_remote_code=True, local_files_only=True
        ).eval()
        self.runtime = self.model[0]
        # ST remote-code loading creates a separate QueryParts class identity.
        # Render with the loaded runtime's helpers, not the entry-point module.
        self.render_groups = importlib.import_module(type(self.runtime).__module__).input_groups
        if not {"choice", "noul", "score"} <= set(self.runtime.tasks):
            raise ValueError("Bekko checkpoint must have Choice, Noul and Score heads")
        self.query_length = self.runtime.query_length if query_length is None else query_length
        self.document_length = (
            self.runtime.document_length if document_length is None else document_length
        )
        # Native empty prediction validates positional capacity without running a forward pass.
        self.model.predict_groups(
            [],
            query_length=self.query_length,
            document_length=self.document_length,
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
            if file.is_file():
                with file.open("rb") as stream:
                    self.file_hashes[str(file.relative_to(self.checkpoint))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        self.revision = hashlib.sha256(
            json.dumps(self.file_hashes, sort_keys=True).encode()
        ).hexdigest()
        self.parameters = parameter_metadata(self.model)

    def predict(self, case: InferenceCase) -> list[Prediction]:
        return self.predict_batch([case])[0]

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        outputs: list[list[Prediction]] = [[] for _ in cases]
        for start in range(0, len(cases), self.case_batch_size):
            groups, references = [], []
            for index, case in enumerate(cases[start : start + self.case_batch_size], start):
                rendered = self.render_groups(
                    native_input(case), prefix_layout=self.runtime.prefix_layout
                )
                for question, group in zip(case.questions, rendered, strict=True):
                    if tuple(o.id for o in question.options) != group.metadata.candidate_ids:
                        raise ValueError("Native candidate order differs from evaluation criteria")
                    groups.append(group)
                    references.append((index, case.case_id, question))
            # Bound preflight tokenization by decisions as well as input cases.
            for offset in range(0, len(groups), self.case_batch_size):
                validate_groups(
                    self.runtime,
                    groups[offset : offset + self.case_batch_size],
                    self.query_length,
                    self.document_length,
                )
            probabilities = self.model.predict_groups(
                groups,
                batch_size=self.case_batch_size,
                token_budget=self.token_budget,
                query_length=self.query_length,
                document_length=self.document_length,
                show_progress_bar=False,
            )
            for (index, case_id, question), values in zip(references, probabilities, strict=True):
                ids = [o.id for o in question.options]
                distribution = dict(zip(ids, values.tolist(), strict=True))
                check_probabilities(distribution, ids)
                outputs[index].append(
                    Prediction(
                        case_id=case_id,
                        question_id=question.id,
                        probabilities=distribution,
                    )
                )
        return outputs

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            **self.parameters,
            id=self.checkpoint.name,
            adapter="bekko-v0",
            revision=self.revision,
            settings={
                "checkpoint_files_sha256": self.file_hashes,
                "checkpoint_encoder_config": self.runtime.settings,
                "query_length": self.query_length,
                "document_length": self.document_length,
                "input_length_policy": "reject-overflow",
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
