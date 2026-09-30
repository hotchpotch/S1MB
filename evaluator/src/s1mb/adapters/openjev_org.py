"""Text decision readout for openjev/openjev, distinct from other Open-Jev models."""

import hashlib
import importlib
import math
import string
from pathlib import Path
from typing import Any

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api, score_options
from .upstream import UpstreamAdapter, state_text

LETTERS = string.ascii_uppercase + string.ascii_lowercase
TEMPERATURE = 0.85
NOUL_TEMPERATURE = 1.829074


def decision_options(question):
    """Translate to native menus, retaining the inverse mapping to authored IDs."""
    native = questions_for_api([question], structured=True, anonymous_choice=True, sort_score=True)[
        question.id
    ]
    instruction = str(native["instructions"])
    criteria = native["criteria"]
    if question.task == "noul":
        ids = ["true", "false"]
        entries = [("yes", criteria["true"]), ("no", criteria["false"])]
    elif question.task == "score":
        ids = [option.id for option in score_options(question, True)]
        entries = [(str(index), value) for index, value in enumerate(criteria)]
        instruction += " Rate along the ordered levels below (lowest first)."
    else:
        ids = [option.id for option in question.options]
        entries = list(criteria.items())
    options = [(key, "" if value is None else state_text(value)) for key, value in entries]
    if question.task == "noul":
        options = [
            (key, description or fallback)
            for (key, description), fallback in zip(
                options, ["The statement is true.", "The statement is false."], strict=True
            )
        ]
    return instruction, options, ids


def render_prompt(state, instruction, options):
    """Use the published text-lane layout, without identifiers or target data."""
    if not 1 <= len(options) <= len(LETTERS):
        raise ValueError("A native readout requires 1 to 52 options")
    menu = "\n".join(
        f"[{letter}] {key}: {description}" for letter, (key, description) in zip(LETTERS, options)
    )
    return (
        f"State:\n{state_text(state)}\n\nQuestion: {instruction}\nOptions:\n{menu}"
        "\n\nAnswer with the letter of the best option only."
    )


def calibrated_probabilities(values, task):
    """Apply the published fixed Noul calibration; retain unrounded precision."""
    if not values or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid readout probabilities")
    if not math.isclose(sum(values), 1, abs_tol=1e-5):
        raise ValueError("Readout probabilities must sum to one")
    if task != "noul":
        return values
    if len(values) != 2:
        raise ValueError("Noul requires two probabilities")
    yes = min(max(values[0], 1e-4), 1 - 1e-4)
    yes = 1 / (1 + math.exp(-math.log(yes / (1 - yes)) / NOUL_TEMPERATURE))
    return [yes, 1 - yes]


class OpenJevOrgAdapter(UpstreamAdapter):
    """Run exact candidate-token logits in BF16 with no generation or truncation."""

    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=16384):
        if not 0 < context_limit <= 16384:
            raise ValueError("OpenJev context_limit must be between 1 and 16384")
        self.setup("openjev-org", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        self.tokenizer: Any = transformers.AutoTokenizer.from_pretrained(self.path)
        encoded = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != 52:
            raise ValueError("Readout letters must be distinct single tokens")
        self.letter_ids = [ids[0] for ids in encoded]
        self.context_limit = context_limit
        self.engine = transformers.AutoModelForImageTextToText.from_pretrained(
            self.path, dtype=self.torch.bfloat16, attn_implementation="sdpa", device_map=device
        ).eval()
        self.attention_model = self.engine
        self.settings = {
            "display_name": "OpenJev 27B (openjev/openjev, BF16)",
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_token_ids": self.letter_ids,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_length": context_limit,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "openjev-org-native-text-pyrepr-anonymous-choice-v1",
            "probability_rule": "exact-letter-logits-native-calibration-unrounded",
            "temperature": TEMPERATURE,
            "noul_temperature": NOUL_TEMPERATURE,
            "noul_bias": 0,
            "noul_clip": [1e-4, 1 - 1e-4],
            "option_overflow": "native-balanced-chunks-winner-anchor-distribution",
            "score_order": "ascending-numeric-values-mapped-back-to-authored-ids",
            "enable_thinking": False,
            "permutations": 1,
            "modalities": ["text"],
            "quantization": None,
        }
        self.enable_kernels()

    def encode(self, state, instruction, options):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": render_prompt(state, instruction, options)}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        if not 0 < len(ids) < self.context_limit:
            raise ValueError(
                f"OpenJev prompt has {len(ids)} tokens; limit {self.context_limit} includes "
                "one readout token; refusing truncation"
            )
        return ids

    def readout(self, state, instruction, options):
        ids = self.encode(state, instruction, options)
        with self.torch.inference_mode():
            inputs = self.torch.tensor([ids], device=self.device)
            logits = (
                self.engine(
                    input_ids=inputs,
                    attention_mask=self.torch.ones_like(inputs),
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits[0, -1, self.letter_ids[: len(options)]]
                .float()
            )
            if not self.torch.isfinite(logits).all():
                raise ValueError("Non-finite candidate logits")
            return (logits / TEMPERATURE).softmax(-1).cpu().tolist()

    def distribution(self, state, instruction, options):
        if len(options) <= 52:
            return self.readout(state, instruction, options)
        count = math.ceil(len(options) / 52)
        if count > 52:
            raise ValueError("Native two-stage readout supports at most 2704 options")
        width = math.ceil(len(options) / count)
        chunks = [options[start : start + width] for start in range(0, len(options), width)]
        parts = [self.readout(state, instruction, chunk) for chunk in chunks]
        winners = [max(range(len(part)), key=part.__getitem__) for part in parts]
        final = self.readout(
            state,
            instruction,
            [chunk[winner] for chunk, winner in zip(chunks, winners, strict=True)],
        )
        values = [
            mass * value / part[winner]
            for mass, part, winner in zip(final, parts, winners, strict=True)
            for value in part
        ]
        total = sum(values)
        return [value / total for value in values]

    def predict(self, case):
        predictions = []
        for question in case.questions:
            instruction, options, ids = decision_options(question)
            values = calibrated_probabilities(
                self.distribution(case.state, instruction, options), question.task
            )
            probabilities = dict(zip(ids, values, strict=True))
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
