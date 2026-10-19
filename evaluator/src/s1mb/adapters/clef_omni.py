"""Clef-Omni's native joint-schema head with CUDA-only thinker placement."""

import importlib
import importlib.util
import sys

from .clef import ClefAdapter
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
        memory = {i: max(0, self.torch.cuda.mem_get_info(i)[0] - 2 * 1024**3) for i in range(2)}
        placement = accelerate.infer_auto_device_map(
            self.engine.thinker, max_memory=memory, dtype=self.torch.bfloat16,
            no_split_module_classes=self.engine.thinker._no_split_modules)
        if any(value not in {0, 1, "cuda:0", "cuda:1"} for value in placement.values()):
            raise ValueError("Clef-Omni weights exceed reserved CUDA memory; CPU/disk offload is disabled")
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
                         "max_input_tokens": context_limit, "input_length_policy": "reject-overflow",
                         "case_batch_size": 1, "questions_per_call": "all-case-questions",
                         "renderer": "native-clef-omni-anonymous-numeric-score-v1",
                         "media": "native-ordered-pil-images-v1",
                         "audio_output": False,
                         "weight_loading": "cpu-staging-then-cuda-only-inference",
                         "image_preprocessing": "shared-full-image-policy-v1-then-native-processor"}
