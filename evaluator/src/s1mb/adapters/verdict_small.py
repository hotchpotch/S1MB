"""Verdict-small's native embedding decisions with verified full-input windows."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, state_text


class VerdictSmallAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("verdict-small", model, revision, source, device)
        native = importlib.import_module("verdict.engines.embed")
        self.schema = importlib.import_module("verdict.types")
        self.engine = native.EmbedEngine(str(self.path), device=device, batch_size=16)
        self.engine.q_prefix, self.engine.p_prefix = native._prefixes_for(model)
        self.context_limit = context_limit or 32768
        self.settings = {
            "dtype": "float32",
            "max_input_tokens": self.context_limit,
            "native_window_tokens": self.engine.model.max_seq_length,
            "input_length_policy": "native-overlapping-state-windows-no-truncation-strict-options",
            "window_pooling": "mean-then-l2-normalize",
            "case_batch_size": 1,
            "input_prefix": self.engine.q_prefix,
            "option_prefix": self.engine.p_prefix,
            "logit_scale": self.engine.scale,
            "temperature": 1.0,
            "renderer": "native-choice-authored-noul-numeric-score-v1",
            "cache_policy": "reset-per-case",
        }

    def predict(self, case):
        self.engine._input_cache.clear()
        self.engine._option_cache.clear()
        tokenizer = self.engine.model.tokenizer
        state = state_text(case.state)
        if len(tokenizer(state, truncation=False)["input_ids"]) > self.context_limit:
            raise ValueError("Verdict-small full state exceeds context limit")
        # Native state chunking retains every token. Check decoded windows again:
        # tokenization plus prefixes must never trigger SentenceTransformer truncation.
        for chunk in self.engine._chunks(state):
            if (
                len(tokenizer(self.engine.q_prefix + chunk, truncation=False)["input_ids"])
                > self.engine.model.max_seq_length
            ):
                raise ValueError("Verdict-small state window exceeds native limit")
        predictions = []
        for question in case.questions:
            item = questions_for_api([question], structured=True, anonymous_choice=True)[
                question.id
            ]
            options = []
            for index, option in enumerate(question.options):
                description = state_text(
                    json.loads(option.description_json)
                    if option.description_json is not None
                    else option.description
                )
                if question.task == "noul":
                    description = f"{option.id}: {description}"
                elif question.task == "score":
                    description = f"{option.value}: {description}"
                options.append(self.schema.Option(label=f"option_{index}", description=description))
            native_question = self.schema.Question(
                kind="choose", prompt=state_text(item["instructions"]), options=options
            )
            for text in self.engine.option_texts(native_question):
                full = self.engine.p_prefix + self.engine.question_text(native_question, text)
                if (
                    len(tokenizer(full, truncation=False)["input_ids"])
                    > self.engine.model.max_seq_length
                ):
                    raise ValueError(
                        "Verdict-small option exceeds native limit; refusing truncation"
                    )
            decision = self.schema.Decision(input=state, question=native_question)
            with self.torch.inference_mode():
                logits = self.engine.score([decision]).logits[0]
            values = self.torch.as_tensor(logits).float().softmax(-1).tolist()
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
