"""Kodiak native calibrated ensemble with complete input and numeric Beta bins."""

import copy
import hashlib
import importlib
import json
import logging
from itertools import pairwise
from typing import Any

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, state_text


class KodiakInputTooLong(ValueError):
    """The native packer would truncate an authored input."""


def kodiak_question(question):
    item = questions_for_api([question], structured=True)[question.id]
    native = {"id": "decision", "text": state_text(item["instructions"]), "allow_null": False}
    if question.task != "score":
        native.update(type="choice", labels=[
            {"id": f"option_{i}", "text": state_text(item["criteria"][option.id])}
            for i, option in enumerate(question.options)
        ])
        return native
    levels = [{"value": option.value, "description": description}
              for option, description in zip(question.options, item["criteria"], strict=True)]
    ordered = sorted(levels, key=lambda level: level["value"])
    native.update(
        type="score", min=ordered[0]["value"], max=ordered[-1]["value"],
        min_label=state_text(ordered[0]["description"]),
        max_label=state_text(ordered[-1]["description"]),
        text=native["text"] + "\nScore levels:\n" + json.dumps(levels, ensure_ascii=False),
    )
    return native


def budget_kodiak_characters(question, question_limit, label_limit):
    """Fit textual fields to native character limits without removing labels."""
    result = copy.deepcopy(question)
    changes = []

    def shorten(value, limit, field):
        if value is None or len(value) <= limit:
            return value
        marker = " […] "
        if limit <= len(marker):
            raise ValueError("Kodiak character budget cannot preserve body text")
        retained = limit - len(marker)
        head = (retained + 1) // 2
        tail = retained // 2
        changes.append((field, len(value), limit))
        return value[:head] + marker + (value[-tail:] if tail else "")

    result["text"] = shorten(result["text"], question_limit, "question")
    for index, label in enumerate(result.get("labels", [])):
        label["text"] = shorten(label["text"], label_limit, f"label_{index}")
    labels = result.get("labels", [])
    normalized = [label["text"].strip().casefold() for label in labels]
    if len(set(normalized)) != len(normalized):
        for index, label in enumerate(labels):
            label["text"] = shorten(
                f"option_{index}: {label['text']}", label_limit, f"label_{index}_distinct",
            )
    for field in ("min_label", "max_label"):
        if field in result:
            result[field] = shorten(result[field], label_limit, field)
    return result, changes


def beta_level_probabilities(question, mean, concentration, cdf):
    ordered = sorted(question.options, key=lambda option: option.value)
    low, high = ordered[0].value, ordered[-1].value
    if low is None or high is None or low >= high:
        raise ValueError("Kodiak Score requires a nondegenerate numeric range")
    if not 0 < mean < 1 or not concentration > 0:
        raise ValueError("Kodiak returned invalid Beta parameters")
    edges = [(left.value + right.value) / 2 for left, right in pairwise(ordered)]
    cumulative = [0.0, *[float(cdf((edge - low) / (high - low),
                                 mean * concentration, (1 - mean) * concentration))
                         for edge in edges], 1.0]
    return {option.id: cumulative[i + 1] - cumulative[i] for i, option in enumerate(ordered)}


class KodiakAdapter(UpstreamAdapter):
    case_batch_size = 1
    native_adaptive_budget = False

    def __init__(self, model, revision, source, device, native_adaptive_budget=False):
        self.native_adaptive_budget = native_adaptive_budget
        self.setup("kodiak", model, revision, source, device)
        self.native = importlib.import_module("kodiak_s1.hub")
        self.infer = importlib.import_module("kodiak_s1.infer")
        self.schema = importlib.import_module("kodiak_s1.schema")
        self.packing = importlib.import_module("kodiak_s1.packing")
        tokenizer_module = importlib.import_module("kodiak_s1.tokenizer")
        spec = json.loads((self.path / "ensemble.json").read_text())
        tokenizer_files = [self.path / member / "tokenizer.json" for member in spec["members"]]
        digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in tokenizer_files]
        if len(set(digests)) != 1:
            raise ValueError("Kodiak ensemble member tokenizers differ")
        tokenizer = importlib.import_module("tokenizers").Tokenizer.from_file(str(tokenizer_files[0]))
        self._tokenizer_bindings: list[tuple[Any, Any]] = [
            (module, module.get_tokenizer) for module in [tokenizer_module, self.packing]
        ]
        for module, _ in self._tokenizer_bindings:
            module.get_tokenizer = lambda: tokenizer
        self.packing._ids_cache.clear()
        try:
            self.engine = self.native.Kodiak.from_pretrained(str(self.path), device=device)
        except BaseException:
            for module, original in self._tokenizer_bindings:
                module.get_tokenizer = original
            self.packing._ids_cache.clear()
            raise
        self.members = self.engine.members
        self.limits = self.packing.Limits()
        self.settings.update({
            "dtype": "native-float32-weights-bfloat16-autocast",
            "attention_implementation": "native-sdpa-with-packed-block-masks",
            "ensemble_members": len(self.members),
            "calibration": [member.calibration for member in self.members],
            "tokenizer_sha256": digests[0], "tokenizer_origin": "bundled-identical-members",
            "native_token_budgets": vars(self.limits), "max_row_tokens": 4096,
            "input_length_policy": "reject-native-truncation-before-forward",
            "allow_null": False, "choice_probability_origin": "native-calibrated-ensemble-conditional",
            "score_probability_origin": "native-moment-matched-beta-numeric-midpoint-bins",
            "renderer": "native-structured-anonymous-numeric-score-v1",
        })

        if self.native_adaptive_budget:
            self.settings.update({
                "input_length_policy": "explicit-native-adaptive-budget-v3",
                "candidate_budget_policy": "preserve-all-candidate-counts;native-description-truncation",
                "native_truncation_enabled": True,
                "character_budget_policy": "head-tail-before-native-validation",
                "duplicate_label_policy": "anonymous-option-prefix-after-shortening",
                "max_question_characters": self.schema.MAX_QUESTION_CHARS,
                "max_label_characters": self.schema.MAX_LABEL_CHARS,
            })

    def predict(self, case):
        predictions = []
        for question in case.questions:
            native = kodiak_question(question)
            if self.native_adaptive_budget:
                native, changes = budget_kodiak_characters(
                    native, self.schema.MAX_QUESTION_CHARS, self.schema.MAX_LABEL_CHARS,
                )
                for field, before, after in changes:
                    logging.getLogger(__name__).warning(
                        "Explicit character budget: case=%s question=%s field=%s characters=%d->%d policy=head-tail-v1",
                        case.case_id, question.id, field, before, after,
                    )
            request = self.schema.Request.model_validate({
                "state": case.state, "questions": [native],
            }).model_dump()
            packed = self.packing.pack_example(request, self.limits, with_targets=False)
            if packed.truncated:
                if not self.native_adaptive_budget:
                    raise KodiakInputTooLong("Native input budgets exceeded; refusing truncation")
                before = 2 + len(self.packing._encode(self.packing.render_state(request["state"])))
                before += 1 + len(self.packing._encode(self.packing.question_text(request["questions"][0])))
                before += sum(1 + len(self.packing._encode(label["text"])) for label in native.get("labels", []))
                logging.getLogger(__name__).warning(
                    "Explicit native budget: case=%s question=%s adapter=kodiak tokens=%d->%d candidates=%d",
                    case.case_id, question.id, before, len(packed), len(question.options))
            if len(packed) > 4096:
                raise KodiakInputTooLong("Native packed row exceeds 4096 tokens")
            raw = self.infer.combine_raw([
                self.infer.raw_outputs(member.model, [request]) for member in self.members
            ])
            if len(raw) != 1 or len(raw[0]) != 1 or raw[0][0]["qid"] != "decision":
                raise ValueError("Kodiak returned incorrect question count")
            answer = raw[0][0]
            if answer["type"] != native["type"] or answer["p_null"] != 0:
                raise ValueError("Kodiak returned incompatible answer type or null mass")
            if question.task == "score":
                probabilities = beta_level_probabilities(
                    question, answer["mu"], answer["kappa"], self.infer.beta_dist.cdf,
                )
            else:
                expected = [label["id"] for label in native["labels"]]
                if answer["labels"] != expected:
                    raise ValueError("Kodiak returned different choice labels")
                probabilities = dict(zip(
                    [option.id for option in question.options], answer["cond_probs"], strict=True,
                ))
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=probabilities,
            ))
        return predictions

    def close(self):
        for module, original in getattr(self, "_tokenizer_bindings", []):
            module.get_tokenizer = original
        if hasattr(self, "packing"):
            self.packing._ids_cache.clear()
        if hasattr(self, "members"):
            del self.members
        super().close()
