"""JEV 27B's released converted LoRA, native wide labels and calibrated readout."""

import importlib
import itertools
import json
import math
import os
import string
import urllib.request

from s1mb.data import ModelInfo, Prediction, check_probabilities
from s1mb.parameters import parameter_metadata

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, state_text


def jev27_probabilities(logprobs, candidates, biases, temperature):
    """Read every requested native label, add its trained bias, then calibrate once."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid JEV 27B temperature")
    values = []
    for token, bias in zip(candidates, biases, strict=True):
        key = f"token_id:{token}"
        if key not in logprobs:
            raise ValueError("JEV 27B backend omitted a requested candidate")
        value = (logprobs[key] + bias) / temperature
        if not math.isfinite(value):
            raise ValueError("JEV 27B backend returned nonfinite probabilities")
        values.append(value)
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def jev27_parameter_metadata(path):
    """Count the native text backbone, unmerged converted LoRA and trained biases."""
    accelerate = importlib.import_module("accelerate")
    transformers = importlib.import_module("transformers")
    config = transformers.AutoConfig.from_pretrained(str(path))
    with accelerate.init_empty_weights(include_buffers=True):
        model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    model.tie_weights()
    counts = parameter_metadata(model)
    safe_open = importlib.import_module("safetensors").safe_open
    extra = 24
    with safe_open(str(path / "adapter_vllm/adapter_model.safetensors"), framework="pt", device="cpu") as tensors:
        keys = tensors.keys()
        for key in keys:
            extra += math.prod(tensors.get_slice(key).get_shape())
    counts["total_params"] += extra
    counts["active_params"] += extra
    return counts


class JEV27Adapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, server_host, server_port, context_limit=None):
        self.setup("jev27", model, revision, source, device)
        config = json.loads((self.path / "judge_config.json").read_text())
        if config["slots"]["template_version"] != "bare-v1" or config["softcap"] is not None:
            raise ValueError("Unsupported released JEV 27B head configuration")
        self.head = json.loads((self.path / "adapter_vllm/decision_head.json").read_text())
        self.temperature = json.loads((self.path / "calibration.json").read_text())["per_kind"]["choice"]
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid native JEV 27B temperature")
        self.limit = 32768 if context_limit is None else context_limit
        if not 1 <= self.limit <= 32768:
            raise ValueError("JEV 27B runtime limit must be within 1..32768")
        self.tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(str(self.path))
        if self.tokenizer is None:
            raise ValueError("Checkpoint did not provide a supported tokenizer")
        self.labels, self.tokens = [], []
        for label in list(string.ascii_uppercase) + ["".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)]:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) == 1 and ids[0] in self.tokenizer.encode(f"x\n{label}) y", add_special_tokens=False):
                self.labels.append(label)
                self.tokens.append(ids[0])
            if len(self.labels) == 256:
                break
        if len(set(self.tokens)) != 256 or self.tokens[:16] != self.head["verbalizer_ids"][8:24]:
            raise ValueError("JEV 27B tokenizer differs from its native readout")
        self.url = f"http://{server_host}:{server_port}"
        with urllib.request.urlopen(self.url + "/v1/models", timeout=30) as response:
            models = json.load(response)
        loaded = {item["id"]: item for item in models["data"]}
        if (set(loaded) != {"base", "jev"} or loaded["base"].get("root") != "/models/jev"
                or loaded["jev"].get("root") != "/models/jev/adapter_vllm"):
            raise ValueError("JEV 27B backend must serve the pinned base and converted LoRA")
        self.counts = jev27_parameter_metadata(self.path)
        self.settings.update({
            "dtype": "bfloat16", "runtime": "vllm-0.31.0", "temperature": self.temperature,
            "max_input_tokens": self.limit, "max_candidates": 256, "thinking": False,
            "choice_strategy": "native-single-pass", "head": "released-converted-lm-head-lora-plus-bias",
            "input_length_policy": "reject-complete-prompt-plus-answer-overflow",
            "renderer": "bare-v1-no-bos-authored-order-numeric-score-choice",
            "candidate_logprobs": "complete-selected-token-readout-no-tail-substitution",
            "runtime_models": models,
            "runtime_image": "vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b",
        })

    def metadata(self):
        return ModelInfo(id=self.model_id, adapter=self.name, revision=self.revision,
                         **self.counts, settings={"device": self.device,
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "source_python_sha256": self.source_digest, **self.settings})

    def predict(self, case):
        predictions = []
        wire = sifr_questions(case)
        for question in case.questions:
            spec = wire[question.id]
            options = [state_text(value) for value in spec["criteria"].values()]
            if not 2 <= len(options) <= 256:
                raise ValueError("JEV 27B requires 2..256 candidates")
            lines = [f"{label}) {value}" for label, value in zip(self.labels[:len(options)], options, strict=True)]
            text = ("[kind] choice\n[state] " + state_text(case.state)
                    + "\n[question] " + state_text(spec["instructions"])
                    + "\n[options]\n" + "\n".join(lines) + "\n[decision]:")
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) + 1 > self.limit:
                raise ValueError("Complete native JEV 27B input exceeds runtime capacity")
            candidates = self.tokens[:len(options)]
            body = {"model": "jev", "prompt": ids, "max_tokens": 1, "temperature": 1.0,
                    "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
                    "add_special_tokens": False,
                    "logprobs": len(options), "allowed_token_ids": candidates,
                    "return_tokens_as_token_ids": True}
            request = urllib.request.Request(self.url + "/v1/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=1800) as response:
                result = json.load(response)
            if result["usage"]["prompt_tokens"] != len(ids) or len(result["choices"]) != 1:
                raise ValueError("JEV 27B backend changed the complete input")
            biases = [self.head["bias"][8 + i] if i < 16 else 0.0 for i in range(len(options))]
            values = jev27_probabilities(result["choices"][0]["logprobs"]["top_logprobs"][0],
                                        candidates, biases, self.temperature)
            probabilities = dict(zip([option.id for option in question.options], values, strict=True))
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions
