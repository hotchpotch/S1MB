"""Open Spark's menu prompt, restricted logits and released calibration."""

import importlib
import json
import string
from typing import Any, cast

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .jevlite import extended_labels
from .upstream import UpstreamAdapter, state_text


class SparkAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("spark", model, revision, source, device)
        native = importlib.import_module("open_spark_jev.model")
        self.prompt = cast(Any, importlib.import_module("open_spark_jev.prompting"))
        self.schema = importlib.import_module("open_spark_jev.schema")
        self.engine = native.MenuScorer(
            str(self.path),
            device=device,
            dtype=self.torch.bfloat16,
            use_state_cache=False,
            attn_implementation="sdpa",
        )
        self.attention_model = self.engine.lm
        self.context_limit = context_limit or 8192
        self.codes = extended_labels(
            self.engine.tokenizer, list(string.ascii_uppercase), max_candidates or 26
        )
        self.prompt.LABEL_CHARS = self.codes
        self.code_ids = [
            self.engine.tokenizer.encode(code, add_special_tokens=False)[0] for code in self.codes
        ]
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit,
            "max_candidates": len(self.codes),
            "native_max_candidates": 26,
            "answer_codes": self.codes,
            "temperature": self.engine.calibration.temperature,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "native-menu-authored-noul-as-choice-numeric-score-v1",
        }
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        # Native State accepts text, objects and arrays; serialization also admits
        # scalar structured state without imposing a lossy native character cap.
        content = state_text(case.state)
        state = self.schema.State(content=content)
        for question in case.questions:
            if len(question.options) > len(self.codes):
                raise ValueError("Spark candidate capacity exceeded; refusing to drop options")
            item = questions_for_api([question], structured=True, anonymous_choice=True)[
                question.id
            ]
            options = []
            for index, option in enumerate(question.options):
                label = (
                    option.id
                    if question.task == "noul"
                    else str(option.value)
                    if question.task == "score"
                    else f"option_{index}"
                )
                description = (
                    json.loads(option.description_json)
                    if option.description_json is not None
                    else option.description
                )
                options.append(f"{label}: {state_text(description)}")
            if question.task == "score":
                native_question = self.schema.Score.model_construct(
                    prompt=state_text(item["instructions"]), levels=options, allow_abstain=False
                )
            else:
                native_question = self.schema.Choice.model_construct(
                    prompt=state_text(item["instructions"]), options=options, allow_abstain=False
                )
            text = self.prompt.render_prefix(
                state, max_chars=max(len(content), 1)
            ) + self.prompt.render_suffix(native_question)
            ids = self.engine.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > self.context_limit:
                raise ValueError("Spark input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                logits = (
                    self.engine.lm(
                        input_ids=self.torch.tensor([ids], device=self.device),
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    .logits[0, -1]
                    .float()
                )
                values = (
                    (
                        logits[self.code_ids[: len(options)]]
                        / self.engine.calibration.t(question.task)
                    )
                    .softmax(-1)
                    .cpu()
                    .tolist()
                )
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
