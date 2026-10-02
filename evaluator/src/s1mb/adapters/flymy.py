"""FlyMy.AI's verified packaged pointer runtime, with explicit context limits."""

import importlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import decision_row
from .jevlite import extended_labels
from .upstream import UpstreamAdapter


def bounded_request_json(value):
    """Retain native JSON validation with a 1 MiB local request byte limit."""
    maximum = 1024 * 1024
    remaining_nodes = 4096

    def validate(item, depth):
        nonlocal remaining_nodes
        remaining_nodes -= 1
        if remaining_nodes < 0 or depth > 24:
            raise ValueError("Request structure exceeds the bound")
        if isinstance(item, dict):
            if len(item) * 2 > remaining_nodes:
                raise ValueError("Request structure exceeds the bound")
            for key in item:
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                validate(key, depth + 1)
                validate(item[key], depth + 1)
        elif isinstance(item, list):
            if len(item) > remaining_nodes:
                raise ValueError("Request structure exceeds the bound")
            for element in item:
                validate(element, depth + 1)
        elif isinstance(item, str):
            if len(item) > maximum:
                raise ValueError("Request text exceeds 1 MiB")
        elif not isinstance(item, (type(None), bool, int, float)):
            raise TypeError("Request must contain JSON values")

    validate(value, 0)
    size = 0
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for fragment in encoder.iterencode(value):
        size += len(fragment.encode("utf-8"))
        if size > maximum:
            raise ValueError("Request exceeds 1 MiB")
    return size


class FlymyAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("flymy", model, revision, source, device)
        # The author's manifest verifier requires real package members, while
        # Hub snapshots intentionally use symlinks into their blob store.
        output = Path.cwd() / "output"
        output.mkdir(exist_ok=True)
        self.package = tempfile.TemporaryDirectory(prefix="flymy-runtime-", dir=output)
        package = Path(self.package.name) / "package"
        shutil.copytree(self.path, package, symlinks=False)
        spec = importlib.util.spec_from_file_location("s1mb_flymy_native", package / "model.py")
        if spec is None or spec.loader is None:
            raise ValueError("Missing FlyMy native model.py loader")
        self.native = cast(Any, importlib.util.module_from_spec(spec))
        sys.modules[spec.name] = self.native
        spec.loader.exec_module(self.native)
        config = json.loads((package / "model.json").read_text())
        letter_readout = hasattr(self.native, "LETTERS")
        capacity = None
        if letter_readout:
            transformers = importlib.import_module("transformers")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config["base_model"],
                revision=config["base_revision"],
            )
            capacity = max_candidates or len(self.native.LETTERS)
            self.native.LETTERS = extended_labels(tokenizer, list(self.native.LETTERS), capacity)
            self.engine = self.native.load(device=device, graphs=False)
            decider = self.engine
            self.attention_model = decider.model
            native_limit = config["max_input_tokens"]
            calibration = {"temperature": decider.temperature}
        else:
            self.engine = self.native.load()
            decider = self.engine._decider
            self.attention_model = decider.model.lm
            native_limit = self.native.SERVING["max_tokens"]
            calibration = decider.calibrator
        self.context_limit = context_limit or decider.max_tokens
        decider.max_tokens = self.context_limit
        if not letter_readout:
            # The native 64 KiB transport cap rejects valid long benchmark text.
            # Keep its structural checks and the decider's strict token bound.
            self.native.bounded_json = bounded_request_json
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit,
            "native_max_input_tokens": native_limit,
            "input_length_policy": "native-strict-reject-overflow",
            "case_batch_size": 1,
            "readout": "native-letter-logits" if letter_readout else "native-pointer",
            "max_candidates": capacity,
            "native_max_candidates": 26 if letter_readout else None,
            "base_model": config["base_model"],
            "base_revision": config["base_revision"],
            "calibration": calibration,
            "package_manifest_verified": True,
            "max_request_bytes": None if letter_readout else 1024 * 1024,
            "native_max_request_bytes": None if letter_readout else 65536,
            "renderer": "native-anonymous-choice-numeric-score-v1",
        }
        self.set_attention("sdpa")
        if letter_readout:
            self.enable_kernels()

    def predict(self, case):
        predictions = []
        for q in case.questions:
            question = decision_row(case.state, q)["question"]
            result = self.engine.decide(case.state, question)
            keys = (
                ["false", "true"]
                if q.task == "noul"
                else list(question["criteria"])
                if q.task == "choice"
                else [str(i) for i in range(len(q.options))]
            )
            raw = result["probabilities"]
            check_probabilities(raw, keys)
            ids = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={key: raw[native] for key, native in zip(ids, keys, strict=True)},
                )
            )
        return predictions

    def close(self):
        try:
            super().close()
        finally:
            self.package.cleanup()
