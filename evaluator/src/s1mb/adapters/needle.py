"""Needle 3 CUDA likelihoods for complete, declared tool-call candidates."""

import gc
import importlib
import importlib.metadata
import json
import math
import os
import re
from collections.abc import Mapping

from s1mb.data import ModelInfo, Prediction, check_probabilities
from s1mb.parameters import METHOD

from .nimble import nimble_field
from .upstream import checkpoint_path, source_path, state_text


def needle_parameter_metadata(params):
    """Count loaded Flax parameters, excluding token and Engram embeddings.

    Embeddings remain excluded when shared with output heads. Include other
    loaded heads, matching embedding_excluded_parameters_v1;
    this is not a count of operations or parameters used by one input.
    """
    sizes = {}
    lookup = set()

    def visit(value, path=()):
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, (*path, str(key)))
            return
        key = id(value)
        sizes[key] = math.prod(value.shape)
        if len(path) == 2 and path[1] == "embedding" and (
            path[0] == "embedding" or re.fullmatch(r"engrams_\d+", path[0])
        ):
            lookup.add(key)

    visit(params)
    if not sizes:
        raise ValueError("No Needle parameters found")
    excluded = lookup
    return {
        "total_params": sum(sizes.values()),
        "active_params": sum(size for key, size in sizes.items() if key not in excluded),
        "parameter_count_method": METHOD,
    }


class NeedleAdapter:
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if device != "cuda:0":
            raise ValueError("Needle requires a single explicitly selected CUDA GPU")
        self.model_id = model
        self.source, self.source_digest = source_path(source)
        self.path, self.revision = checkpoint_path(model, revision)
        self.jax = importlib.import_module("jax")
        self.jax.config.update("jax_platforms", "cuda")
        devices = self.jax.devices("gpu")
        if len(devices) != 1:
            raise ValueError("Expose exactly one GPU for Needle; CPU fallback is forbidden")
        self.device = devices[0]
        self.jnp = importlib.import_module("jax.numpy")
        runtime = importlib.import_module("needle.model.run")
        tokens = importlib.import_module("needle.model.tokenizer")
        self.renderer = importlib.import_module("needle.model.finetune")
        params, config = runtime.load_checkpoint(str(self.path / "checkpoints/needle3.safetensors"))
        self.parameter_counts = needle_parameter_metadata(params)
        original_limit = config.max_seq_len
        self.context_limit = context_limit or original_limit
        config.max_seq_len = self.context_limit
        self.engine = runtime.SimpleAttentionNetwork(config)
        self.params = self.jax.device_put(params, self.device)
        self.tokenizer = tokens.SANTokenizer(str(self.path / "tokenizer/tokenizer.model"))
        self.bos, self.eos, self.pad = tokens.BOS_ID, tokens.EOS_ID, tokens.PAD_ID

        @self.jax.jit
        def likelihood(params, ids, target_start, target_end):
            logits = self.engine.apply({"params": params}, ids)[0, :-1].astype(self.jnp.float32)
            logprobs = self.jax.nn.log_softmax(logits, axis=-1)
            chosen = self.jnp.take_along_axis(logprobs, ids[0, 1:, None], axis=-1)[:, 0]
            positions = self.jnp.arange(ids.shape[1] - 1) + 1
            return self.jnp.where(
                (positions >= target_start) & (positions < target_end), chosen, 0
            ).sum()

        self.likelihood = likelihood
        self.settings = {
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_python_sha256": self.source_digest,
            "dtype": config.dtype,
            "attention_implementation": "jax-native",
            "max_input_tokens": self.context_limit,
            "checkpoint_runtime_limit": original_limit,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "probability_origin": "conditional-complete-tool-call-sequence-likelihood",
            "native_confidence_used": False,
            "length_normalization": False,
            "renderer": "native-record-decision-anonymous-choice-numeric-score-v1",
            "versions": {
                name: importlib.metadata.version(name) for name in ["jax", "flax", "sentencepiece"]
            },
        }

    def metadata(self):
        return ModelInfo(
            id=self.model_id,
            adapter="needle",
            revision=self.revision,
            settings=self.settings,
            **self.parameter_counts,
        )

    def predict(self, case):
        predictions = []
        for question in case.questions:
            keys, schema = nimble_field(question)
            definition = schema["decision"]
            tool = {
                "name": "record_decision",
                "description": "Record the answer to this question about the text: "
                + definition["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": keys,
                            "description": definition["description"]
                            + " Options: "
                            + json.dumps(definition["choice_descriptions"], ensure_ascii=False),
                        }
                    },
                    "required": ["decision"],
                },
            }
            rows = []
            for key in keys:
                prompt, target = self.renderer.render_example(
                    {
                        "query": state_text(case.state),
                        "system": definition["description"],
                        "tools": [tool],
                        "answers": [{"name": "record_decision", "arguments": {"decision": key}}],
                    }
                )
                prefix = [self.bos] + self.tokenizer.encode(prompt)
                ids = prefix + self.tokenizer.encode(target) + [self.eos]
                if len(ids) > self.context_limit:
                    raise ValueError(
                        "Needle complete candidate exceeds context limit; refusing truncation"
                    )
                rows.append((ids, len(prefix)))
            scores = []
            for ids, start in rows:
                width = min(self.context_limit, ((len(ids) + 127) // 128) * 128)
                array = self.jax.device_put(
                    self.jnp.array([ids + [self.pad] * (width - len(ids))], dtype=self.jnp.int32),
                    self.device,
                )
                scores.append(
                    float(self.likelihood(self.params, array, start, len(ids)).block_until_ready())
                )
            values = self.jax.nn.softmax(self.jnp.array(scores)).tolist()
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions

    def close(self):
        del self.params
        self.jax.clear_caches()
        gc.collect()
