"""OpenDecision's native typed NLI interfaces with strict tokenization."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, state_text


class LiteralHypothesisClassifier:
    """Keep authored braces literal while retaining the final label placeholder."""

    def __init__(self, classifier):
        self.classifier = classifier

    def __getattr__(self, name):
        return getattr(self.classifier, name)

    def __call__(self, *args, **kwargs):
        template = kwargs.get("hypothesis_template")
        if template is not None:
            prefix, marker, suffix = template.rpartition("{}")
            if not marker:
                raise ValueError("Native hypothesis template has no label placeholder")

            def escape(value):
                return value.replace("{", "{{").replace("}", "}}")

            kwargs["hypothesis_template"] = escape(prefix) + "{}" + escape(suffix)
        return self.classifier(*args, **kwargs)


class StrictTokenizer:
    """Retain native tokenization while refusing every requested truncation."""

    def __init__(self, tokenizer, limit):
        self.tokenizer = tokenizer
        self.limit = limit

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        kwargs["truncation"] = False
        kwargs.pop("max_length", None)
        encoded = self.tokenizer(*args, **kwargs)
        rows = encoded["input_ids"]
        if hasattr(rows, "shape"):
            largest = rows.shape[-1]
        elif rows and isinstance(rows[0], int):
            largest = len(rows)
        else:
            largest = max((len(row) for row in rows), default=0)
        if largest > self.limit:
            raise ValueError("OpenDecision input exceeds context limit; refusing truncation")
        return encoded


class OpenDecisionAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("opendecision", model, revision, source, device)
        native = importlib.import_module("opendecision.engine")
        self.engine = native.OpenDecisionEngine(model=str(self.path), device=device, batch_size=1)
        self.attention_model = self.engine.classifier.model
        self.context_limit = context_limit or 8192
        self.engine.classifier.tokenizer = StrictTokenizer(
            self.engine.classifier.tokenizer, self.context_limit
        )
        self.engine.classifier = LiteralHypothesisClassifier(self.engine.classifier)
        self.settings = {
            "dtype": str(next(self.attention_model.parameters()).dtype),
            "max_input_tokens": self.context_limit,
            "input_length_policy": "reject-overflow",
            "checkpoint_position_limit": self.attention_model.config.max_position_embeddings,
            "case_batch_size": 1,
            "candidate_batch_size": 1,
            "renderer": "native-typed-authored-noul-numeric-score-v1",
            "noul_hypothesis": "instruction-plus-authored-definition",
            "hypothesis_brace_policy": "literal-authored-braces-final-label-placeholder",
        }
        self.set_attention("sdpa")

    def predict(self, case):
        predictions = []
        for question in case.questions:
            item = questions_for_api([question], structured=True, anonymous_choice=True)[
                question.id
            ]
            instruction = state_text(item["instructions"])
            if question.task == "score":
                criteria = [
                    f"level_{i}: {o.value}: {state_text(json.loads(o.description_json) if o.description_json is not None else o.description)}"
                    for i, o in enumerate(question.options)
                ]
                answer = self.engine.score(
                    state=case.state, instructions=instruction, criteria=criteria
                )
                keys = [str(i) for i in range(len(criteria))]
            elif question.task == "noul":
                criteria = {
                    key: f"{instruction}\n\n{key}: {state_text(value)}"
                    for key, value in item["criteria"].items()
                }
                answer = self.engine.noul(
                    state=case.state, instructions=instruction, criteria=criteria
                )
                value = answer["noul"]
                probabilities = {"true": value, "false": 1 - value}
                keys = []
            else:
                criteria = {key: state_text(value) for key, value in item["criteria"].items()}
                answer = self.engine.choice(
                    state=case.state, instructions=instruction, criteria=criteria
                )
                keys = list(criteria)
            if question.task != "noul":
                check_probabilities(answer["probabilities"], keys)
                probabilities = {
                    o.id: answer["probabilities"][key]
                    for o, key in zip(question.options, keys, strict=True)
                }
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
