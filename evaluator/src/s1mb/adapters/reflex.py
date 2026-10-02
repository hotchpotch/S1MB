"""Reflex's trained adapter, native isolated branches and calibrated readout."""

import importlib
import json
import string
from typing import Any, cast

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import decision_row
from .jevlite import extended_labels
from .upstream import UpstreamAdapter, checkpoint_path, state_text


def normalize_native_probabilities(probabilities, keys):
    """Restore unit mass within Reflex's documented six-decimal rounding bound."""
    if set(probabilities) != set(keys):
        raise ValueError("Reflex probability keys differ from the request")
    if any(not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("Invalid Reflex probability")
    total = sum(probabilities.values())
    if not total > 0 or abs(total - 1) > len(keys) * 0.5e-6 + 1e-12:
        raise ValueError("Reflex probability sum exceeds six-decimal rounding error")
    normalized = {key: value / total for key, value in probabilities.items()}
    check_probabilities(normalized, keys)
    return normalized


class ReflexAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("reflex", model, revision, source, device)
        native = importlib.import_module("reflex.engine")
        self.prompt = cast(Any, importlib.import_module("reflex.prompt"))
        self.schema = cast(Any, importlib.import_module("reflex.schema"))
        config = json.loads((self.path / "adapter_config.json").read_text())
        base = config["base_model_name_or_path"]
        base_path, base_revision = checkpoint_path(base, config.get("revision") or "main")
        self.context_limit = context_limit or 8192
        self.engine = native.Engine.load(
            str(base_path),
            adapter_path=str(self.path),
            device=device,
            dtype=self.torch.bfloat16,
            attn_implementation="sdpa",
            max_pack_tokens=self.context_limit,
            max_branch_tokens=self.context_limit,
            state_cache_entries=0,
            default_permutations=1,
        )
        self.attention_model = self.engine.model
        self.codes = extended_labels(
            self.engine.tok, list(string.ascii_uppercase), max_candidates or 26
        )
        self.prompt.LETTERS = self.codes
        self.schema.MAX_CHOICE_OPTIONS = len(self.codes)
        self.schema.MAX_SCORE_LEVELS = len(self.codes)
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "base_model": base,
            "base_revision": base_revision,
            "max_input_tokens": self.context_limit,
            "max_candidates": len(self.codes),
            "native_max_candidates": 26,
            "answer_codes": self.codes,
            "temperature": self.engine.cal.temperature,
            "calibration_head": self.engine.cal.head,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "state_cache_entries": 0,
            "permutations": 1,
            "probability_rounding_decimals": 6,
            "probability_normalization": "bounded-native-rounding-error",
            "renderer": "native-branches-anonymous-choice-numeric-score-v1",
        }
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for question in case.questions:
            row = decision_row(case.state, question)
            request = self.schema.SystemOneRequest(
                state=case.state
                if isinstance(case.state, (str, dict, list))
                else state_text(case.state),
                questions={"decision": row["question"]},
                permutations=1,
            )
            native_question = request.questions["decision"]
            branches = self.prompt.build_branches("decision", native_question, self.engine.fmt, 1)
            prefix_ids = self.engine.tok.encode(
                self.engine.fmt.prefix(request.state), add_special_tokens=False
            )
            if any(
                len(prefix_ids) + len(self.engine.tok.encode(branch.text, add_special_tokens=False))
                > self.context_limit
                for branch in branches
            ):
                raise ValueError("Reflex input exceeds context limit; refusing truncation")
            answer = self.engine.answer(request).answers["decision"]
            if question.task == "noul":
                probabilities = {"true": answer.noul, "false": 1 - answer.noul}
            else:
                keys = (
                    list(row["question"]["criteria"])
                    if question.task == "choice"
                    else [str(i) for i in range(len(question.options))]
                )
                native_probabilities = normalize_native_probabilities(answer.probabilities, keys)
                probabilities = {
                    o.id: native_probabilities[key]
                    for o, key in zip(question.options, keys, strict=True)
                }
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
