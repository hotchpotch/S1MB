"""Pinned DM-JEPA latent readout with lossless state and option encoding."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, checkpoint_path

BACKBONE = "answerdotai/ModernBERT-base"
BACKBONE_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


def dm_jepa_inputs(case, question, formatter):
    """Retain native formatting without targets, identifiers or provenance."""
    definition = questions_for_api([question], structured=True)[question.id]
    state = formatter.format_state_prompt(case.state, definition["instructions"])
    texts = []
    for index, option in enumerate(question.options):
        value = (
            json.loads(option.description_json)
            if option.description_json is not None
            else option.description
        )
        criteria = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        label = (
            option.id
            if question.task == "noul"
            else str(option.value)
            if question.task == "score"
            else f"option_{index}"
        )
        texts.append(formatter.format_option_prompt(label, criteria))
    return state, texts


def encode_inputs(tokenizer, state, options, state_limit, option_limit):
    """Tokenize intact before any GPU transfer, retaining native special tokens."""
    state_batch = tokenizer(state, truncation=False, return_tensors="pt")
    option_batch = tokenizer(options, padding=True, truncation=False, return_tensors="pt")
    if state_batch["input_ids"].shape[-1] > state_limit:
        raise ValueError("DM-JEPA state exceeds context limit; refusing truncation")
    if option_batch["input_ids"].shape[-1] > option_limit:
        raise ValueError("DM-JEPA criterion exceeds option limit; refusing truncation")
    return state_batch, option_batch


class DMJEPAAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, device, context_limit=None):
        if not device.startswith("cuda"):
            raise ValueError("DM-JEPA requires explicit CUDA; no CPU fallback")
        if context_limit is not None and not 1 <= context_limit <= 16384:
            raise ValueError("DM-JEPA context_limit must be between 1 and 16384")
        path, resolved = checkpoint_path(model, revision)
        device = "cuda:0" if device == "cuda" else device
        self.setup("dm-jepa", model, resolved, str(path), device)
        config = json.loads((path / "config.json").read_text())
        if config["backbone_model"] != BACKBONE:
            raise ValueError("Unsupported DM-JEPA backbone")
        hub = importlib.import_module("huggingface_hub")
        backbone_path = hub.snapshot_download(
            BACKBONE,
            revision=BACKBONE_REVISION,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        )
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(backbone_path)
        native = importlib.import_module("djepa.model.djepa")
        self.formatter = importlib.import_module("djepa.dataset.formatter").JevFormatter
        self.engine = native.DJEPA(
            model_name=backbone_path,
            pretrained=False,
            **config["latent_predictor"],
            init_temperature=config["scorer"]["init_temperature"],
            gradient_checkpointing=False,
        )
        weights = importlib.import_module("safetensors.torch").load_file(
            str(path / "model.safetensors")
        )
        self.engine.load_state_dict(weights, strict=True)
        self.engine = self.engine.to(device).eval()
        self.attention_model = self.engine.state_encoder.backbone
        self.set_attention("sdpa")
        self.state_limit = min(config["max_state_length"], context_limit or 16384)
        self.option_limit = config["max_option_length"]
        self.settings.update(
            {
                "dtype": "float32-weights-bfloat16-autocast",
                "max_input_tokens": self.state_limit,
                "max_option_tokens": self.option_limit,
                "input_length_policy": "reject-overflow",
                "renderer": "native-independent-state-options-numeric-score-v1",
                "questions_per_call": 1,
                "backbone_id": BACKBONE,
                "backbone_revision": BACKBONE_REVISION,
                "backbone_max_position_embeddings": self.attention_model.config.max_position_embeddings,
                "context_condition": "release-state-budget",
                "source_revision": resolved,
                "native_logit_scale": float(self.engine.scorer.temperature.detach().cpu()),
            }
        )

    def predict(self, case):
        results = []
        for question in case.questions:
            state, options = dm_jepa_inputs(case, question, self.formatter)
            state_batch, option_batch = encode_inputs(
                self.tokenizer, state, options, self.state_limit, self.option_limit
            )
            with (
                self.torch.inference_mode(),
                self.torch.autocast("cuda", dtype=self.torch.bfloat16),
            ):
                output = self.engine(
                    state_input_ids=state_batch["input_ids"].to(self.device),
                    state_attention_mask=state_batch["attention_mask"].to(self.device),
                    option_input_ids=option_batch["input_ids"].unsqueeze(0).to(self.device),
                    option_attention_mask=option_batch["attention_mask"]
                    .unsqueeze(0)
                    .to(self.device),
                )
            values = output["probabilities"][0].float().cpu().tolist()
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            results.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return results
