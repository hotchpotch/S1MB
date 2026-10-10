"""Eikos FP8 native semif rendering with complete candidate log probabilities."""

import importlib
import json
import math
import os
import urllib.request

from s1mb.data import ModelInfo, Prediction, check_probabilities
from s1mb.parameters import parameter_metadata

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, state_text


def eikos_fp8_probabilities(rows, candidates, temperature):
    """Reject missing label probabilities instead of substituting a guessed logit."""
    scores = {}
    for row in rows:
        token = row["token"]
        if not token.startswith("token_id:"):
            raise ValueError("EikosFP8 backend must return token IDs")
        key = int(token.removeprefix("token_id:"))
        if key in scores:
            raise ValueError("EikosFP8 returned duplicate token probabilities")
        scores[key] = row["logprob"]
    if any(key not in scores for key in candidates):
        raise ValueError("EikosFP8 backend omitted requested candidate log probabilities")
    values = [scores[key] / temperature for key in candidates]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("EikosFP8 returned nonfinite candidate log probabilities")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def eikos_fp8_parameter_metadata(path):
    """Count logical native parameters, excluding FP8 quantization scale buffers."""
    accelerate = importlib.import_module("accelerate")
    transformers = importlib.import_module("transformers")
    config = transformers.AutoConfig.from_pretrained(str(path))
    with accelerate.init_empty_weights(include_buffers=True):
        model = transformers.Qwen3_5ForConditionalGeneration(config)
    # Accelerate's parameter-registration hook prevents aliases inside its context.
    model.tie_weights()
    return parameter_metadata(model)


class EikosFP8Adapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, server_host, server_port, context_limit=None):
        self.setup("eikos-fp8", model, revision, source, device)
        config = json.loads((self.path / "decision_config.json").read_text())
        if config["prompt_version"] != "letter-v1-semif" or config["readout"] != "letter-logit":
            raise ValueError("Unsupported Eikos FP8 native decision protocol")
        self.calibration = json.loads((self.path / "calib.json").read_text())
        self.native = importlib.import_module("decision_core")
        if self.native.STYLE != "semif":
            raise ValueError("Eikos FP8 requires native semif prompt rendering")
        self.limit = 32768 if context_limit is None else context_limit
        if not 1 <= self.limit <= 32768:
            raise ValueError("Eikos FP8 runtime context limit must be within 1..32768")
        self.tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(str(self.path))
        labels = [self.tokenizer.encode(label, add_special_tokens=False) for label in self.native.LABELS]
        if any(len(ids) != 1 for ids in labels):
            raise ValueError("Native Eikos labels must be single tokens")
        self.tokens = [ids[0] for ids in labels]
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Native Eikos labels must have distinct token IDs")
        self.url = f"http://{server_host}:{server_port}"
        with urllib.request.urlopen(self.url + "/v1/models", timeout=30) as response:
            models = json.load(response)
        if [item["id"] for item in models["data"]] != ["decider"]:
            raise ValueError("EikosFP8 backend must serve the pinned model as decider")
        if models["data"][0].get("root") != "/models/eikos":
            raise ValueError("EikosFP8 backend must load the verified snapshot at /models/eikos")
        self.counts = eikos_fp8_parameter_metadata(self.path)
        self.settings.update({
            "quantization": "compressed-tensors-FP8-dynamic",
            "calibration": self.calibration, "max_input_tokens": self.limit,
            "input_length_policy": "reject-overflow-including-answer-token",
            "thinking": False, "runtime": "vllm-0.31.0",
            "runtime_image": "vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b",
            "renderer": "native-letter-v1-semif-authored-order-numeric-score",
            "candidate_logprobs": "explicit-all-token-ids-no-top-k-substitution",
            "runtime_models": models,
        })

    def metadata(self):
        return ModelInfo(id=self.model_id, adapter=self.name, revision=self.revision,
                         **self.counts, settings={"device": self.device,
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "source_python_sha256": self.source_digest, **self.settings})

    def predict(self, case):
        wire = sifr_questions(case)
        predictions = []
        for question in case.questions:
            count = len(question.options)
            if not 2 <= count <= self.native.MAX_ONE_PASS:
                raise ValueError("Eikos FP8 candidate capacity exceeded")
            definition = wire[question.id]
            definition["instructions"] = state_text(definition["instructions"])
            definition["criteria"] = {key: state_text(value) for key, value in definition["criteria"].items()}
            options = self.native.options_of(definition)
            prompt = self.tokenizer.apply_chat_template(
                self.native.messages(case.state, definition, options),
                add_generation_prompt=True, enable_thinking=False, tokenize=False,
            )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + 1 > self.limit:
                raise ValueError("Complete Eikos FP8 prompt exceeds context limit")
            candidates = self.tokens[:count]
            payload = {
                "model": "decider", "prompt": ids, "max_tokens": 1,
                "temperature": 0, "logprobs": count,
                "allowed_token_ids": candidates, "return_tokens_as_token_ids": True,
            }
            request = urllib.request.Request(
                self.url + "/v1/completions", json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=1800) as response:
                result = json.load(response)
            if result["usage"]["prompt_tokens"] != len(ids):
                raise ValueError("EikosFP8 backend tokenization differs from the complete native prompt")
            top = result["choices"][0]["logprobs"]["top_logprobs"][0]
            rows = [{"token": token, "logprob": value} for token, value in top.items()]
            temperature = self.native.temp_for(self.calibration, 1.0, len(ids), count, "choice")
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Invalid Eikos FP8 calibration temperature")
            values = eikos_fp8_probabilities(rows, candidates, temperature)
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions
