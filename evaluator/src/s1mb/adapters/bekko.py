"""Evaluate local Bekko Sentence Transformers models through their native API."""

import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from s1mb.data import InferenceCase, ModelInfo, Prediction
from s1mb.parameters import parameter_metadata

from .input_lengths import (
    check_balanced_input_lengths_batched,
    check_input_lengths,
)


def native_input(case: InferenceCase) -> dict:
    """Reconstruct only the structured inference fields for the training renderer."""
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


class BekkoAdapter:
    case_batch_size = 128
    token_budget = 64_000

    def __init__(
        self,
        checkpoint: str,
        source: str,
        device: str,
        query_length: int = 16384,
        document_length: int = 2048,
        *,
        token_budget: int = 64_000,
    ):
        if query_length < 3:
            raise ValueError("query_length must be at least 3")
        if document_length < 2:
            raise ValueError("document_length must be at least 2")
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.token_budget = token_budget
        if not device.startswith("cuda"):
            raise ValueError("Bekko evaluation requires CUDA")
        sys.path.insert(0, str(Path(source).resolve() / "src"))
        self.bekko = importlib.import_module("bekko_system_one")
        self.release = importlib.import_module("bekko_system_one.release")
        self.torch = importlib.import_module("torch")
        self.torch.set_num_threads(4)
        st = importlib.import_module("sentence_transformers")
        self.model = st.SentenceTransformer(
            checkpoint, device=device, trust_remote_code=True, local_files_only=True
        ).eval()
        if not {"choice", "noul", "score"} <= set(self.model[1].tasks):
            raise ValueError("Bekko checkpoint must have Choice, Noul and Score heads")
        self.checkpoint_encoder_config = dict(self.model[0].settings)
        self.model[0].query_length = query_length
        self.model[0].settings["query_length"] = query_length
        self.model[0].document_length = document_length
        self.model[0].settings["document_length"] = document_length
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.file_hashes = {}
        for path in sorted(self.checkpoint.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    self.file_hashes[str(path.relative_to(self.checkpoint))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        self.revision = hashlib.sha256(
            json.dumps(self.file_hashes, sort_keys=True).encode()
        ).hexdigest()
        self.source_hashes = {
            str(path.relative_to(Path(source) / "src")): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((Path(source) / "src/bekko_system_one").rglob("*.py"))
        }

    def predict(self, case: InferenceCase) -> list[Prediction]:
        return self.predict_batch([case])[0]

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        groups, references = [], []
        for index, case in enumerate(cases):
            source = native_input(case)
            for position, question in enumerate(case.questions):
                groups.append(self.release.render_input_group(source, position))
                references.append((index, case.case_id, question))
        check_input_lengths(
            self.model[0].tokenizer,
            [g.query for g in groups],
            [d for g in groups for d in g.candidates],
            self.model[0].query_length,
            self.model[0].document_length,
        )
        if self.model[0].settings.get("query_truncation") == "balanced":
            check_balanced_input_lengths_batched(
                self.model[0].tokenizer,
                [g.query_parts for g in groups],
                self.model[0].query_length,
            )
        probabilities = self._predict_batched_transfer(groups)
        outputs: list[list[Prediction]] = [[] for _ in cases]
        for (index, case_id, question), values in zip(references, probabilities, strict=True):
            outputs[index].append(
                Prediction(
                    case_id=case_id,
                    question_id=question.id,
                    probabilities=dict(
                        zip(
                            [option.id for option in question.options], values.tolist(), strict=True
                        )
                    ),
                )
            )
        return outputs

    def _predict_batched_transfer(self, groups):
        """Keep native packing and softmax, but transfer once per microbatch."""
        data = importlib.import_module("bekko_system_one.data")
        inference = importlib.import_module("bekko_system_one.inference")
        with self.torch.inference_mode():
            was_training = self.model.training
            try:
                self.model.eval()
                runtime = inference.inference_runtime(self.model, "optimized")
                prepared = data.prepare_groups(groups, self.model[0])
                results, batch = [], []

                def flush():
                    scores = runtime.score(data.collate_groups(batch, self.model[0]))
                    lengths = [len(group.documents) for group in batch]
                    probabilities = [s.float().softmax(0) for s in scores.flatten().split(lengths)]
                    results.extend(self.torch.cat(probabilities).cpu().split(lengths))

                for group in prepared:
                    if batch and data.work_tokens([*batch, group]) > self.token_budget:
                        flush()
                        batch = []
                    batch.append(group)
                if batch:
                    flush()
                return results
            finally:
                self.model.train(was_training)

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            **parameter_metadata(self.model),
            id=f"{self.checkpoint.parent.parent.name}-{self.checkpoint.parent.name}",
            adapter="bekko",
            revision=self.revision,
            settings={
                "checkpoint_files_sha256": self.file_hashes,
                "source_files_sha256": self.source_hashes,
                "encoder_config": self.model[0].settings,
                "checkpoint_encoder_config": self.checkpoint_encoder_config,
                "input_length_policy": "reject-overflow",
                "tasks": list(self.model[1].tasks),
                "device": self.device,
                "inference": "optimized",
                "case_batch_size": self.case_batch_size,
                "microbatch_tokens": self.token_budget,
                "prediction_pipeline": "batched-transfer-v1",
                "renderer": "native-training-input-v2",
                "prefix_layout": "instruction_state",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "temperature": 1.0,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "torch",
                        "transformers",
                        "sentence-transformers",
                        "flash-attn",
                        "datasets",
                    ]
                },
            },
        )

    def close(self) -> None:
        self.bekko.clear_inference_cache(self.model)
        del self.model
        self.torch.cuda.empty_cache()
