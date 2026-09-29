"""Optional bridge to an explicitly supplied System Ichi checkout and checkpoint."""

import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

from s1mb.data import InferenceCase, ModelInfo, Prediction
from s1mb.parameters import parameter_metadata

from .input_lengths import check_input_lengths


class SystemIchiAdapter:
    case_batch_size = 128
    question_batch_tokens = 32_768

    def __init__(
        self,
        checkpoint: str,
        source: str,
        device: str,
        questions_per_call: int | None = None,
        query_length: int | None = None,
        document_length: int | None = None,
    ):
        if questions_per_call is not None and questions_per_call < 1:
            raise ValueError("questions_per_call must be positive")
        if query_length is not None and query_length < 3:
            raise ValueError("query_length must be at least 3")
        if document_length is not None and document_length < 2:
            raise ValueError("document_length must be at least 2")
        self.strict_input_lengths = query_length is not None or document_length is not None
        self.questions_per_call = questions_per_call
        if not device.startswith("cuda"):
            raise ValueError("Optimized System Ichi inference requires CUDA")
        sys.path.insert(0, str(Path(source).resolve() / "src"))
        module = importlib.import_module("system_ichi_exp.typed_decision.model")
        self.batching = importlib.import_module("system_ichi_exp.typed_decision.batching")
        self.torch = importlib.import_module("torch")
        self.model = module.TypedDecisionModel.load(checkpoint).to(device).eval()
        self.render_decision = module.render_decision
        if query_length is not None:
            self.model.encoder.query_length = query_length
        if document_length is not None:
            self.model.encoder.document_length = document_length
        optimization = importlib.import_module("system_ichi_exp.reranker.cuda_inference")
        # Optimize only the encoder snapshot; task heads retain their FP32 weights.
        self.model.encoder = optimization.prepare_shared_inference(
            self.torch.nn.ModuleList([self.model.encoder]).eval()
        )[0]
        assert optimization.__file__ is not None
        self.optimization_digest = hashlib.sha256(
            Path(optimization.__file__).read_bytes()
        ).hexdigest()
        self.checkpoint, self.device = checkpoint, device
        self.checkpoint_hashes = {
            str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(checkpoint).rglob("*"))
            if p.is_file()
        }
        self.revision = hashlib.sha256(
            json.dumps(self.checkpoint_hashes, sort_keys=True).encode()
        ).hexdigest()
        self.encoder_config = json.loads(
            (Path(checkpoint) / "encoder/ranking_config.json").read_text()
        )
        assert module.__file__ is not None
        assert self.batching.__file__ is not None
        self.batching_digest = hashlib.sha256(Path(self.batching.__file__).read_bytes()).hexdigest()
        self.source_digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()

    def predict(self, case: InferenceCase) -> list[Prediction]:
        return self.predict_batch([case])[0]

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        items, references = [], []
        for index, case in enumerate(cases):
            state = json.dumps(
                case.state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            for q in case.questions:
                items.append(
                    {
                        "state_json": state,
                        "decision": {
                            "decision_id": q.id,
                            "type": q.task,
                            "options": [o.model_dump() for o in q.options],
                            "target": None,
                        },
                        "prompt": {
                            "decision_id": q.id,
                            "system_prompt": q.system_prompt,
                            "instruction": q.instructions,
                        },
                    }
                )
                references.append((index, case.case_id, q))
        if not items:
            return [[] for _ in cases]
        if self.strict_input_lengths:
            rendered = [self.render_decision(item) for item in items]
            check_input_lengths(
                self.model.encoder.tokenizer,
                [q for q, _, _ in rendered],
                [d for _, docs, _ in rendered for d in docs],
                self.model.encoder.query_length,
                self.model.encoder.document_length,
            )
        prepared = self.batching.prepare_decisions(items, self.model.encoder)
        positions = {id(item): i for i, item in enumerate(prepared)}
        device_probabilities = {}
        with (
            self.torch.inference_mode(),
            self.torch.autocast(
                "cuda",
                dtype=self.torch.bfloat16,
            ),
        ):
            for group in self._question_groups(prepared):
                logits = self.model.forward_prepared(group)
                # One FP32 softmax for all independent candidate distributions.
                # Padding receives zero probability and is removed before export.
                padded = self.torch.nn.utils.rnn.pad_sequence(
                    logits, batch_first=True, padding_value=float("-inf")
                )
                probabilities = padded.float().softmax(-1)
                for index, item in enumerate(group):
                    device_probabilities[positions[id(item)]] = probabilities[
                        index, : len(item.documents)
                    ]
            # Transfer all small probability vectors once after the GPU work.
            sizes = [len(item.documents) for item in prepared]
            probabilities = (
                self.torch.cat([device_probabilities[i] for i in range(len(prepared))])
                .cpu()
                .split(sizes)
            )
            distributions = [values.tolist() for values in probabilities]
        result: list[list[Prediction]] = [[] for _ in cases]
        for (index, case_id, q), probabilities in zip(references, distributions, strict=True):
            result[index].append(
                Prediction(
                    case_id=case_id,
                    question_id=q.id,
                    probabilities=dict(zip([o.id for o in q.options], probabilities, strict=True)),
                )
            )
        return result

    def _question_groups(self, prepared):
        """Pack complete questions across cases under a bounded token budget."""
        limit = self.questions_per_call or max(1, len(prepared))
        for start in range(0, len(prepared), limit):
            yield from self.batching.pack_decisions(
                prepared[start : start + limit],
                self.question_batch_tokens,
                padded_documents=True,
            )

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            **parameter_metadata(self.model),
            id=Path(self.checkpoint).parent.name,
            adapter="system-ichi",
            revision=self.revision,
            settings={
                "source_module_sha256": self.source_digest,
                "checkpoint_files_sha256": self.checkpoint_hashes,
                "encoder_config": self.encoder_config,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ["torch", "transformers", "sentence-transformers", "flash-attn"]
                },
                "device": self.device,
                "query_length": self.model.encoder.query_length,
                "document_length": self.model.encoder.document_length,
                "renderer": "reviewed-prompt-v1",
                "input_length_policy": "reject-overflow"
                if self.strict_input_lengths
                else "checkpoint-truncation",
                "temperature": 1.0,
                "inference_path": "prepared-fused-cross-case-batches-v1",
                "case_batch_size": self.case_batch_size,
                "questions_per_call": self.questions_per_call,
                "question_batch_scope": "cross-case",
                "question_batch_tokens": self.question_batch_tokens,
                "head_path": "native-forward-prepared",
                "softmax": "padded-fp32",
                "optimization_source_sha256": self.optimization_digest,
                "batching_source_sha256": self.batching_digest,
            },
        )

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
