"""GLiNER2 schema classification using the native full softmax distribution."""

import importlib
import json
from typing import Any, cast

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, state_text


def gliner_inputs(state, question):
    item = questions_for_api([question], structured=True, anonymous_choice=True)[question.id]
    labels = {}
    for index, option in enumerate(question.options):
        key = (
            option.id
            if question.task == "noul"
            else f"level_{index}: {option.value}"
            if question.task == "score"
            else f"option_{index}"
        )
        labels[key] = state_text(
            json.loads(option.description_json)
            if option.description_json is not None
            else option.description
        )
    return f"Question: {state_text(item['instructions'])}\n\n{state_text(state)}", labels


class Gliner2Adapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("gliner2", model, revision, source, device)
        native = importlib.import_module("gliner2")
        self.engine = (
            native.AutoExtractor.from_pretrained(str(self.path), use_flashdeberta=True)
            .to(device)
            .eval()
        )
        self.attention_model = self.engine.encoder
        self.flash_bias: Any = None
        if type(self.engine.encoder).__name__ == "FlashDebertaV2Model":
            self.flash_bias = cast(
                Any, importlib.import_module("flashdeberta.ops.flash_attention_bias")
            )
            self.original_bias_config = self.flash_bias.get_fwd_config
            # The native FP32 heuristic can exceed Blackwell's shared-memory
            # limit for 512--1024 tokens. Use its supported runtime tile override.
            self.flash_bias.get_fwd_config = lambda *_args: (32, 32, 1, 4)
        self.context_limit = context_limit or 8192
        self.settings = {
            "dtype": "float32",
            "encoder_class": type(self.engine.encoder).__name__,
            "attention_implementation": (
                "flashdeberta"
                if type(self.engine.encoder).__name__ == "FlashDebertaV2Model"
                else self.engine.encoder.config._attn_implementation
            ),
            "max_input_tokens": self.context_limit,
            "checkpoint_position_limit": getattr(
                self.engine.encoder.config, "max_position_embeddings", None
            ),
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "probability_origin": "native-classification-softmax-all-labels",
            "renderer": "native-schema-anonymous-choice-numeric-score-v1",
        }
        if self.flash_bias is not None:
            self.settings["flash_bias_tile"] = {
                "block_m": 32,
                "block_n": 32,
                "num_stages": 1,
                "num_warps": 4,
            }

    def close(self):
        if self.flash_bias is not None:
            self.flash_bias.get_fwd_config = self.original_bias_config
        super().close()

    def predict(self, case):
        predictions = []
        for question in case.questions:
            text, labels = gliner_inputs(case.state, question)
            schema = self.engine.create_schema().classification(
                "decision",
                labels,
                multi_label=True,
                cls_threshold=0.0,
                class_act="softmax",
            )
            record = self.engine.processor.transform_record(text, schema, max_len=None)
            if len(record.input_ids) > self.context_limit:
                raise ValueError("GLiNER2 input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                result = self.engine.extract(text, schema, include_confidence=True, max_len=None)
            raw = {row["label"]: row["confidence"] for row in result["decision"]}
            check_probabilities(raw, list(labels))
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=question.id,
                    probabilities={
                        o.id: raw[key] for o, key in zip(question.options, labels, strict=True)
                    },
                )
            )
        return predictions
