"""JADE's released converted decision head and native chunked candidate readout."""

import importlib
import json
import math
import os
import urllib.request

from s1mb.data import ModelInfo, Prediction, check_probabilities
from s1mb.parameters import parameter_metadata

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, checkpoint_path


def jade_candidate_logits(chunks, candidates):
    """Gather all requested native codes without estimating any missing probability."""
    values = {}
    for requested, top in chunks:
        for token in requested:
            key = f"token_id:{token}"
            if key not in top:
                raise ValueError("JADE backend omitted a requested candidate probability")
            value = top[key]
            if token in values or not math.isfinite(value):
                raise ValueError("JADE returned duplicate or nonfinite candidate probabilities")
            values[token] = value
    if set(values) != set(candidates):
        raise ValueError("JADE returned an incomplete candidate set")
    return [values[token] for token in candidates]


def jade_parameter_metadata(base_path, release):
    """Count the native text model and every loaded exported LoRA parameter."""
    transformers = importlib.import_module("transformers")
    accelerate = importlib.import_module("accelerate")
    config = transformers.AutoConfig.from_pretrained(str(base_path))
    with accelerate.init_empty_weights(include_buffers=True):
        model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    model.tie_weights()
    counts = parameter_metadata(model)
    extra = 0
    safe_open = importlib.import_module("safetensors").safe_open
    with safe_open(str(release / "vllm-adapter/adapter_model.safetensors"), framework="pt", device="cpu") as tensors:
        keys = tensors.keys()
        for key in keys:
            extra += math.prod(tensors.get_slice(key).get_shape())
    counts["total_params"] += extra
    counts["active_params"] += extra
    return counts


class JadeAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, server_host, server_port):
        self.setup("jade", model, revision, source, device)
        self.native = importlib.import_module("jade.engine")
        self.prompt = importlib.import_module("jade.decision_prompt")
        manifest = self.native.validate_release(self.source)
        expected = json.loads((self.path / "release-manifest.json").read_text())
        if manifest != expected:
            raise ValueError("JADE source does not match the pinned release")
        self.config = json.loads((self.source / "decision_config.json").read_text())
        base, base_revision = checkpoint_path(self.config["base_model"], self.config["revision"])
        self.tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(str(base))
        self.tokens = self.config["token_ids"]
        codes = self.config["codes"]
        if (len(set(self.tokens)) != 255 or len(codes) != 255
                or [self.tokenizer.encode(code, add_special_tokens=False) for code in codes]
                != [[token] for token in self.tokens]):
            raise ValueError("JADE decision vocabulary does not match the pinned base tokenizer")
        self.temperature = float(self.config["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid JADE native temperature")
        self.url = f"http://{server_host}:{server_port}"
        with urllib.request.urlopen(self.url + "/v1/models", timeout=30) as response:
            models = json.load(response)
        loaded = {item["id"]: item for item in models["data"]}
        if (set(loaded) != {"base", "jade"} or loaded["base"].get("root") != "/models/jade-base"
                or loaded["jade"].get("root") != "/models/jade/vllm-adapter"):
            raise ValueError("JADE backend must load the verified base and converted adapter")
        self.counts = jade_parameter_metadata(base, self.source)
        self.settings = {
            "dtype": "bfloat16", "runtime": "vllm-0.31.0",
            "base_model": self.config["base_model"], "base_revision": base_revision,
            "temperature": self.temperature, "max_input_tokens": 8192,
            "max_candidates": 255, "logprob_token_chunk_size": 128,
            "input_length_policy": "native-reject-overflow-including-answer-token",
            "head": "released-converted-vllm-lora-rank256",
            "language_model_only": True, "thinking": False,
            "typed_mapping": "authored-order-explicit-choice-noul-and-numeric-score",
            "renderer": "native-jade-decision-prompt",
            "runtime_models": models, "release_id": manifest["release_id"],
        }

    def metadata(self):
        return ModelInfo(id=self.model_id, adapter=self.name, revision=self.revision,
                         **self.counts, settings={"device": self.device,
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "source_python_sha256": self.source_digest, **self.settings})

    def predict(self, case):
        predictions = []
        questions = sifr_questions(case)
        for question in case.questions:
            definition = questions[question.id]
            keys, _ = self.prompt.options(definition)
            if not 1 <= len(keys) <= 255:
                raise ValueError("JADE requires 1..255 candidates")
            rendered = self.tokenizer.apply_chat_template(
                self.prompt.decision_messages({"state": case.state, "question": definition}, self.config["codes"]),
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            if len(ids) + 1 > 8192:
                raise ValueError("Complete JADE prompt exceeds native 8192-token capacity")
            candidates = self.tokens[:len(keys)]
            chunks = []
            for offset in range(0, len(candidates), 128):
                part = candidates[offset:offset + 128]
                body = {"model": "jade", "prompt": ids, "max_tokens": 1, "temperature": 0,
                        "logprobs": 0, "logprob_token_ids": part, "allowed_token_ids": candidates,
                        "return_tokens_as_token_ids": True}
                request = urllib.request.Request(self.url + "/v1/completions", json.dumps(body).encode(),
                                                 {"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=1800) as response:
                    result = json.load(response)
                if result["usage"]["prompt_tokens"] != len(ids) or len(result["choices"]) != 1:
                    raise ValueError("JADE backend changed the complete native input")
                chunks.append((part, result["choices"][0]["logprobs"]["top_logprobs"][0]))
            probabilities = self.native.distribution(jade_candidate_logits(chunks, candidates), self.temperature)
            raw = {option.id: value for option, value in zip(question.options, probabilities, strict=True)}
            check_probabilities(raw, [option.id for option in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id, probabilities=raw))
        return predictions
