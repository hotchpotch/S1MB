"""Clef-Omni's native joint-schema head with CUDA-only thinker placement."""

import importlib
import importlib.util
import sys

from .clef import ClefAdapter
from .cuda_sharding import two_gpu_cut
from .upstream import checkpoint_path


class ClefOmniAdapter(ClefAdapter):
    def __init__(self, model, revision, source, device, context_limit=32768, cuda_shards=2):
        if cuda_shards != 2 or device != "cuda:0":
            raise ValueError("Clef-Omni requires two CUDA shards and primary cuda:0")
        path, resolved = checkpoint_path(model, revision)
        self.setup("clef-omni", model, resolved, str(path), device)
        if self.torch.cuda.device_count() != 2:
            raise ValueError("Clef-Omni requires exactly two visible CUDA devices")
        spec = importlib.util.spec_from_file_location(f"s1mb_clef_omni_{resolved}", path / "joint_schema_model.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load pinned Clef-Omni runtime")
        self.native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.native
        spec.loader.exec_module(self.native)
        self.engine, self.processor = self.native.load_release_model(
            path, device="cpu", dtype=self.torch.bfloat16, attn_implementation="sdpa")
        # No native forward is called until every parameter has reached CUDA.
        self.engine.head.to(device)
        self.engine.output_embeddings.to(device)
        accelerate = importlib.import_module("accelerate")
        thinker = self.engine.thinker
        if set(dict(thinker.named_children())) != {"audio_tower", "visual", "model", "lm_head"} or set(
            dict(thinker.model.named_children())
        ) != {"embed_tokens", "layers", "norm", "rotary_emb"}:
            raise ValueError("Unsupported native Clef-Omni thinker structure")
        sizes = accelerate.utils.compute_module_sizes(thinker)
        names = [f"model.layers.{i}" for i in range(len(thinker.model.layers))]
        layers = [sizes[name] for name in names]
        reserve = 3 * 1024**3 // 2
        free = [self.torch.cuda.mem_get_info(i)[0] for i in range(2)]
        cut = two_gpu_cut(layers, sizes[""] - sum(layers), free, activation_reserve=reserve)
        # Explicit complete layers avoid automatic placement's extra swap reserve.
        # Never map an ancestor: dispatch would transiently preload its subtree.
        placement = {
            "audio_tower": 0, "visual": 0, "lm_head": 0,
            "model.embed_tokens": 0, "model.norm": 0, "model.rotary_emb": 0,
            **{name: int(i >= cut) for i, name in enumerate(names)},
        }
        accelerate.dispatch_model(self.engine.thinker, device_map=placement, main_device=0, force_hooks=True)
        self.engine.thinker.register_forward_hook(
            lambda module, inputs, output: accelerate.utils.send_to_device(output, device))
        if any(parameter.device.type != "cuda" for parameter in self.engine.parameters()):
            raise RuntimeError("Clef-Omni contains non-CUDA parameters after dispatch")
        self.attention_model = self.engine.thinker
        self.context_limit = context_limit
        self.image_receipts = {}
        self.settings = {"dtype": "bfloat16", "attention_implementation": "sdpa",
                         "cuda_shards": 2, "cuda_device_map": placement,
                         "activation_reserve_bytes_per_device": reserve,
                         "max_input_tokens": context_limit, "input_length_policy": "reject-overflow",
                         "case_batch_size": 1, "questions_per_call": "all-case-questions",
                         "renderer": "native-clef-omni-anonymous-numeric-score-v1",
                         "media": "native-ordered-pil-images-v1",
                         "audio_output": False,
                         "weight_loading": "cpu-staging-then-cuda-only-inference",
                         "image_preprocessing": "shared-full-image-policy-v1-then-native-processor"}
