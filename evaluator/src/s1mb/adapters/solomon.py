"""Solomon's native question-side LoRA and trained semantic heads."""

import importlib
import json
import tempfile
from pathlib import Path

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, checkpoint_path, state_text
from .vision_native import ordered_images


def solomon_options(definition):
    """Keep numeric/structured criteria and distinguish equal descriptions by position."""
    values = [state_text(value) for value in definition["criteria"].values()]
    if len(set(values)) != len(values):
        values = [f"option_{i}: {value}" for i, value in enumerate(values)]
    return values


class SolomonAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, precision="bf16", cuda_shards=1):
        if precision not in {"bf16", "int8"} or cuda_shards not in {1, 2}:
            raise ValueError("Solomon supports bf16/int8 and one/two CUDA devices")
        if cuda_shards == 2 and precision != "int8":
            raise ValueError("Two-GPU Solomon currently requires native int8 loading")
        self.setup("solomon", model, revision, source, device)
        serving = importlib.import_module("solomon.serving")
        self.native = importlib.import_module("solomon.engine_contract")
        service = importlib.import_module("solomon.service")
        manifest = json.loads((self.path / "MANIFEST.json").read_text())
        base = manifest["base_weights_referenced_not_redistributed"][0]
        base_path, base_revision = checkpoint_path(base["repo"], base["revision"])
        self.engine = serving.ServiceEngine(
            str(self.path / "adapter/adapter.safetensors"),
            str(self.path / "adapter/heads.npz"),
            expected_adapter=manifest["adapter_sha256"],
            expected_heads=json.loads((self.path / "adapter/config.json").read_text())["heads_sha256"],
            precision=precision, model_dir=str(base_path),
        )
        self.store = tempfile.TemporaryDirectory(prefix="s1mb-solomon-")
        self.service = service.service(
            self.store.name, self.engine, selection=self.path / "serving/selection.json",
            binding=self.path / ("serving/serving-binding-int8.json" if precision == "int8" else "serving/serving-binding.json"),
            evidence_head=self.path / "serving/evidence-head",
            evidence_thresholds=self.path / "serving/evidence-policy.json",
        )
        # Native LoRA tensors and numpy heads are otherwise absent from module counts.
        for module in list(self.engine.model.modules()):
            if type(module).__name__ == "LoRA":
                for name in ("a", "b"):
                    setattr(module, name, self.torch.nn.Parameter(
                        getattr(module, name), requires_grad=False,
                    ))
        self.engine.head_metadata = self.torch.nn.ParameterList([
            self.torch.nn.Parameter(self.torch.from_numpy(value), requires_grad=False)
            for head in self.engine.answer_heads.values() for value in head.values()
        ])
        placement = None
        if cuda_shards == 2:
            if self.torch.cuda.device_count() != 2:
                raise ValueError("Solomon sharding requires exactly two visible CUDA devices")
            backbone = self.engine.model.model
            if set(dict(backbone.named_children())) != {"visual", "language_model"}:
                raise ValueError("Unsupported Solomon vision backbone structure")
            tower = backbone.language_model
            if set(dict(tower.named_children())) != {"embed_tokens", "layers", "norm", "rotary_emb"}:
                raise ValueError("Unsupported Solomon text tower structure")
            # Native int8 loading first fits the decoder on CUDA 0. Move half of
            # its already quantized layers directly to CUDA 1; never offload.
            cut = len(tower.layers) // 2
            placement = {"visual": 0, "language_model.embed_tokens": 0,
                         "language_model.norm": 0, "language_model.rotary_emb": 0,
                         **{f"language_model.layers.{i}": int(i >= cut) for i in range(len(tower.layers))}}
            accelerate = importlib.import_module("accelerate")
            accelerate.dispatch_model(backbone, device_map=placement, main_device=0, force_hooks=True)
        if any(parameter.device.type != "cuda" for parameter in self.engine.model.parameters()):
            raise RuntimeError("Solomon contains non-CUDA model weights")
        self.image_receipts = {}
        self.settings = {
            "dtype": precision + "-backbone-float32-question-lora-and-heads",
            "cuda_shards": cuda_shards, "cuda_device_map": placement,
            "author_condition_difference": "two-GPU native-int8 layer placement" if cuda_shards == 2 else None,
            "image_preprocessing": "shared-full-image-policy-v1-then-native-processor",
            "attention_implementation": "sdpa",
            "base_model": base["repo"], "base_revision": base_revision,
            "runtime_identity": dict(self.engine.identity),
            "lora_placement": "question-only-native-cached-prefix",
            "max_input_tokens": 32768,
            "input_length_policy": "reject-full-prompt-overflow-before-prefill",
            "candidate_limit": 8,
            "typed_mapping": "explicit-choice-criteria-noul;ordered-numeric-score",
            "choice_head": "single/choiceR", "score_head": "ordered/choiceS",
            "temperature": 1.0,
        }

    supports_images = True

    def predict(self, case):
        images, visual_state, receipts = ordered_images(case)
        if not images:
            return self._predict_parts(case, [{"text": state_text(case.state)}], [])
        self.image_receipts[case.case_id] = receipts
        with tempfile.TemporaryDirectory(dir=self.store.name) as image_dir:
            parts = [{"text": state_text(visual_state)}]
            for i, image in enumerate(images, 1):
                path = Path(image_dir) / f"image-{i}.png"
                image.save(path, format="PNG")
                parts.extend([{"text": f"<image:{i}>"}, {"image": str(path)}])
            return self._predict_parts(case, parts, images)

    def _predict_parts(self, case, parts, images):
        wire = sifr_questions(case)
        prepared = []
        for question in case.questions:
            definition = wire[question.id]
            ordered = question.task == "score"
            block, width = self.native.listwise_block(
                state_text(definition["instructions"]), solomon_options(definition),
                ordered=ordered, reserved=not ordered,
            )
            text = self.engine._render(parts, block)
            if images:
                encoded = self.engine.processor(text=[text], images=images, return_tensors="pt", add_special_tokens=False)
                length = encoded["input_ids"].shape[-1]
            else:
                length = len(self.engine.t.encode(text, add_special_tokens=False))
            if length > 32768:
                raise ValueError("Complete native Solomon prompt exceeds 32768 tokens")
            prepared.append((question, block, width, ordered))
        state = self.engine.prefill(parts)
        predictions = []
        for question, block, width, ordered in prepared:
            result = self.engine.ask(
                state, block, width, execution="cached",
                head_key="ordered/choiceS" if ordered else "single/choiceR",
            )
            # Native single-choice R includes two reserved outcomes; the serving
            # contract slices them out and normalizes only the listed candidates.
            logits = self.torch.as_tensor(result["letter_logits"][:len(question.options)])
            probabilities = self.torch.softmax(logits.double(), dim=-1).tolist()
            raw = {option.id: p for option, p in zip(question.options, probabilities, strict=True)}
            check_probabilities(raw, [option.id for option in question.options])
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=raw,
            ))
        return predictions

    def close(self):
        if hasattr(self, "service"):
            del self.service
        if hasattr(self, "store"):
            self.store.cleanup()
        super().close()
