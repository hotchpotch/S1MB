"""Blink NVFP4 decisions v1 with complete selected-token log probabilities."""

import importlib
import itertools
import json
import math
import os
import string
import urllib.request

from s1mb.data import ModelInfo, Prediction, check_probabilities
from s1mb.parameters import parameter_metadata

from .rune import rune_messages
from .sifr import sifr_questions
from .upstream import UpstreamAdapter


def blink_probabilities(rows, candidates, temperature):
    """Reject missing label probabilities instead of substituting a guessed logit."""
    scores = {}
    for row in rows:
        token = row["token"]
        if not token.startswith("token_id:"):
            raise ValueError("Blink backend must return token IDs")
        key = int(token.removeprefix("token_id:"))
        if key in scores:
            raise ValueError("Blink returned duplicate token probabilities")
        scores[key] = row["logprob"]
    if any(key not in scores for key in candidates):
        raise ValueError("Blink backend omitted requested candidate log probabilities")
    values = [scores[key] / temperature for key in candidates]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Blink returned nonfinite candidate log probabilities")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def blink_parameter_metadata(path):
    """Count logical native parameters; packed quantization buffers are excluded."""
    accelerate = importlib.import_module("accelerate")
    transformers = importlib.import_module("transformers")
    config = transformers.AutoConfig.from_pretrained(str(path))
    with accelerate.init_empty_weights(include_buffers=True):
        model = transformers.Gemma4ForConditionalGeneration(config)
    # Accelerate's parameter-registration hook prevents aliases inside its context.
    model.tie_weights()
    return parameter_metadata(model)


class BlinkAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, server_host, server_port, context_limit=None):
        self.setup("blink", model, revision, source, device)
        config = json.loads((self.path / "decision_config.json").read_text())
        if config["protocol"] != "surogate decisions v1":
            raise ValueError("Unsupported Blink decision protocol")
        self.temperature = config["recommended_decision_temperature"]
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid Blink calibration temperature")
        self.limit = 32768 if context_limit is None else context_limit
        if not 1 <= self.limit <= 32768:
            raise ValueError("Blink runtime context limit must be within 1..32768")
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        self.codes = list(string.ascii_uppercase) + [
            "".join(parts) for parts in itertools.product(string.ascii_uppercase, repeat=2)
        ]
        self.url = f"http://{server_host}:{server_port}"
        with urllib.request.urlopen(self.url + "/v1/models", timeout=30) as response:
            models = json.load(response)
        if [item["id"] for item in models["data"]] != ["blink"]:
            raise ValueError("Blink backend must serve the pinned model as blink")
        if models["data"][0].get("root") != "/models/blink":
            raise ValueError("Blink backend must load the verified snapshot at /models/blink")
        self.counts = blink_parameter_metadata(self.path)
        self.settings.update({
            "quantization": "modelopt_fp4", "kv_cache_dtype": "fp8",
            "temperature": self.temperature, "max_input_tokens": self.limit,
            "input_length_policy": "reject-overflow-including-answer-token",
            "thinking": False, "runtime": "vllm-0.31.0",
            "runtime_image": "vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b",
            "renderer": "surogate-decisions-v1-authored-noul-numeric-score",
            "candidate_logprobs": "explicit-all-token-ids-no-top-k-substitution",
            "candidate_logprob_chunk_size": 128,
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
            if not 2 <= count <= 255:
                raise ValueError("Blink requires 2..255 candidates")
            messages = rune_messages(case.state, wire[question.id], self.codes[:count])
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False, tokenize=False,
            )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + 1 > self.limit:
                raise ValueError("Complete Blink prompt exceeds context limit")
            candidates = []
            for code in self.codes[:count]:
                continuation = self.tokenizer.encode(prompt + code, add_special_tokens=False)
                if continuation[:-1] != ids or len(continuation) != len(ids) + 1:
                    raise ValueError("Blink code is not a single continuation token")
                candidates.append(continuation[-1])
            if len(set(candidates)) != count:
                raise ValueError("Blink codes have duplicate continuation tokens")
            rows = []
            for offset in range(0, count, 128):
                candidate_chunk = candidates[offset:offset + 128]
                payload = {
                    "model": "blink", "messages": messages, "max_tokens": 1,
                    "temperature": 0, "logprobs": True, "top_logprobs": len(candidate_chunk),
                    "logprob_token_ids": candidate_chunk, "return_tokens_as_token_ids": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                request = urllib.request.Request(
                    self.url + "/v1/chat/completions", json.dumps(payload).encode(),
                    {"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=1800) as response:
                    result = json.load(response)
                if result["usage"]["prompt_tokens"] != len(ids):
                    raise ValueError("Blink backend tokenization differs from the complete native prompt")
                chunk_rows = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                rows.extend(row for row in chunk_rows
                            if int(row["token"].removeprefix("token_id:")) in candidate_chunk)
            values = blink_probabilities(rows, candidates, self.temperature)
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions
