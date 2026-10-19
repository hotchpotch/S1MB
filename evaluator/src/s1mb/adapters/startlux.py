"""StartLux Instant's native decision pipeline on CUDA-only Transformers."""

import asyncio
import base64
import importlib
import io

from .base import decode_answers
from .upstream import UpstreamAdapter
from .vision_native import ordered_images, typed_questions


class StartLuxAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768, cuda_shards=1):
        if not device.startswith("cuda"):
            raise ValueError("StartLux requires explicit CUDA; no CPU fallback")
        if context_limit < 1:
            raise ValueError("Context limit must be positive")
        self.setup("startlux", model, revision, source, device)
        if cuda_shards not in {1, 2} or (cuda_shards == 2 and self.torch.cuda.device_count() != 2):
            raise ValueError("StartLux requires one or two visible CUDA devices")
        config_module = importlib.import_module("startlux_decide.modelcfg")
        folder = config_module.load_model_folder(str(self.path))
        if not folder.bidirectional:
            raise ValueError("This adapter requires an Instant bidirectional checkpoint")
        transformers = importlib.import_module("transformers")
        tokenizer = config_module.load_tokenizer(str(self.path))
        native = importlib.import_module("startlux_decide.backends.transformers")
        self.backend = native.TransformersBackend(str(self.path), tokenizer, True, device=device)
        config = transformers.AutoConfig.from_pretrained(str(self.path))
        backbone = getattr(transformers, config.architectures[0]).from_pretrained(
            str(self.path), dtype=self.torch.bfloat16,
            device_map={"": "cpu" if cuda_shards == 2 else device}, attn_implementation="sdpa",
        ).eval()
        self.settings = {"dtype": "bfloat16", "cuda_shards": cuda_shards,
                         "max_input_tokens": context_limit, "input_length_policy": "reject-overflow-native",
                         "renderer": "native-startlux-pipeline-anonymous-choice-numeric-score-v1",
                         "temperature_by_type": folder.temperature_by_type,
                         "prompt_attention": "bidirectional", "thinking": False,
                         "media": "native-ordered-base64-images-v1",
                         "image_preprocessing": "shared-full-image-policy-v1-then-native-processor"}
        if cuda_shards == 2:
            from .cuda_sharding import dispatch_vision_decoder
            backbone.get_output_embeddings().to(device)
            self.settings["cuda_device_map"] = dispatch_vision_decoder(backbone.model, self.torch)
            self.settings["weight_loading"] = "cpu-staging-then-cuda-only-inference"
        self.backend.model = backbone
        self.backend.device = self.torch.device(device)
        self.backend.body = backbone.model.language_model
        self.backend.head = backbone.get_output_embeddings().weight
        self.backend.processor = transformers.AutoProcessor.from_pretrained(str(self.path))
        self.backend.reads_images = True
        self.engine = backbone
        self.attention_model = backbone
        self.enable_kernels()
        # Native parity checks run only after all weights have reached CUDA.
        bidirectional = importlib.import_module("startlux_decide.backends.bidirectional")
        self.backend.check = bidirectional.enable(self.backend.body)
        self.settings["bidirectional_check"] = self.backend.check
        pipeline = importlib.import_module("startlux_decide.pipeline")
        self.decider = pipeline.Decider(folder, self.backend, context_length=context_limit)
        self.image_receipts = {}
        self.loop = asyncio.new_event_loop()

    def predict(self, case):
        images, state, receipts = ordered_images(case)
        encoded = []
        for image in images:
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            encoded.append("data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii"))
        if images:
            self.image_receipts[case.case_id] = receipts
        answers = {}
        for key, question in typed_questions(case).items():
            result = self.loop.run_until_complete(self.decider.decide(
                state if images else case.state, {key: question}, images=encoded or None,
            ))
            if set(result.answers) != {key}:
                raise ValueError("StartLux returned incorrect question alignment")
            answers[key] = result.answers[key]
        return decode_answers(case, answers, anonymous_choice=True)
