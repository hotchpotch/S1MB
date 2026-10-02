"""Deem's published Torch readout and native typed prompt, without truncation."""

import importlib
import sys
from typing import Any, cast

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import decision_row
from .jevlite import extended_labels
from .upstream import UpstreamAdapter


class DeemAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("deem", model, revision, source, device)
        sys.path.insert(0, str(self.source / "serve"))
        self.native = importlib.import_module("deem_server")
        self.format = cast(Any, importlib.import_module("deem.format"))
        backend = self.native.TorchBackend(str(self.path), device=device, batch_size=1)
        self.engine = backend
        self.attention_model = backend.model
        self.context_limit = context_limit or 8192
        codes = extended_labels(
            backend.tokenizer, list(self.format.LETTERS[:26]), max_candidates or 26
        )
        self.format.letter_for_index = lambda index: codes[index]
        backend.max_letters = len(codes)
        backend._letter_ids = self.torch.tensor(
            [backend.tokenizer.encode(code, add_special_tokens=False)[0] for code in codes],
            device=device,
        )
        native_forward = backend.slot_logits

        def checked_forward(prompts, n_valids):
            if any(
                len(backend.tokenizer.encode(prompt, add_special_tokens=False)) > self.context_limit
                for prompt in prompts
            ):
                raise ValueError("Deem input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                return native_forward(prompts, n_valids)

        backend.slot_logits = checked_forward
        self.core = self.native.DeemCore(backend, n_orders=1)
        self.settings = {
            "dtype": "bfloat16",
            "max_input_tokens": self.context_limit,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "temperature": 1.0,
            "calibration": "native-default-no-checkpoint-calibration-file",
            "max_candidates": len(codes),
            "native_max_candidates": 26,
            "answer_codes": codes,
            "option_orderings": 1,
            "renderer": "native-anonymous-choice-numeric-score-v1",
        }
        self.set_attention("sdpa")
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for question in case.questions:
            row = decision_row(case.state, question)
            result = self.core.decide(case.state, {"decision": row["question"]})["answers"][
                "decision"
            ]
            if question.task == "noul":
                value = result["noul"]
                probabilities = {"true": value, "false": 1 - value}
            else:
                raw = result["probabilities"]
                keys = (
                    list(row["question"]["criteria"])
                    if question.task == "choice"
                    else [str(i) for i in range(len(question.options))]
                )
                check_probabilities(raw, keys)
                probabilities = {
                    option.id: raw[key] for option, key in zip(question.options, keys, strict=True)
                }
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
