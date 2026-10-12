"""Pinned Xor compatibility-server scoring through its released SGLang runtime."""

import importlib.util
import json
import os
import urllib.request
from typing import Any, cast

from s1mb.data import ModelInfo, Prediction, check_probabilities
from s1mb.hub_parameters import resolve_counts

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


class XorAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, server_host, server_port, context_limit=None):
        self.setup("xor", model, revision, source, device)
        limit = 131072 if context_limit is None else context_limit
        if not 1 <= limit <= 131072:
            raise ValueError("Xor context limit must be within 1..131072")
        self.url = f"http://{server_host}:{server_port}"
        with urllib.request.urlopen(self.url + "/get_model_info", timeout=30) as response:
            runtime = json.load(response)
        if runtime.get("model_path") != "/models/xor":
            raise ValueError("Xor backend must load the verified checkpoint at /models/xor")
        self.counts = resolve_counts(model, self.revision)
        spec = importlib.util.spec_from_file_location("_s1mb_xor_native", self.source / "xn_base_server.py")
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load Xor's released compatibility server")
        self.native = cast(Any, importlib.util.module_from_spec(spec))
        values = {"XN_MODEL_PATH": str(self.path), "XN_MODEL_ID": model,
                  "XN_SGLANG_URL": self.url, "XN_MAX_CONTEXT": str(limit),
                  "XN_TEMP_JSON": json.dumps({"choice": 1.95, "noul": 1.0, "score": 1.0})}
        previous = {key: os.environ.get(key) for key in values}
        try:
            os.environ.update(values)
            spec.loader.exec_module(self.native)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.settings.update({
            "dtype": "bfloat16", "max_input_tokens": limit,
            "input_length_policy": "reject-overflow-native", "max_candidates": 255,
            "temperature": 1.95, "renderer": "native-choice-authored-noul-numeric-score",
            "runtime": "released-sglang-logprob-fix", "runtime_model_info": runtime,
            "runtime_image": "prakhar1611/xor-sglang@sha256:94c48d2a6cc98dc456cf93f723707ea7dd81dddfe1061e823b348d68bbe8158f",
            "questions_per_call": 1,
        })

    def metadata(self):
        return ModelInfo(id=self.model_id, adapter=self.name,
                         revision=self.counts["revision"],
                         total_params=self.counts["total_params"],
                         active_params=self.counts["active_params"],
                         parameter_count_method=self.counts["parameter_count_method"],
                         settings={"device": self.device,
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "source_python_sha256": self.source_digest, **self.settings})

    def predict(self, case):
        wire = sifr_questions(case)
        predictions = []
        for question in case.questions:
            answers, _, _ = self.native.evaluate(case.state, {"decision": wire[question.id]})
            if set(answers) != {"decision"}:
                raise ValueError("Xor returned incorrect answer count")
            probabilities = answers["decision"]["probabilities"]
            keys = list(wire[question.id]["criteria"])
            check_probabilities(probabilities, keys)
            aligned = dict(zip([o.id for o in question.options],
                               [probabilities[k] for k in keys], strict=True))
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=aligned))
        return predictions
