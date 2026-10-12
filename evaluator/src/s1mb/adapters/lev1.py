"""Pinned Lev1 packed/cached label scoring with native option-order averaging."""

import importlib

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter


class Lev1Adapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("Lev1 context_limit must be positive")
        self.setup("lev1", model, revision, source, device)
        self.native = importlib.import_module("lev1.engine")
        self._sdpa_flags = {
            name: getattr(self.torch.backends.cuda, f"{flag}_sdp_enabled")()
            for name, flag in self.native.SDPA_FLAGS.items()
        }
        self.engine = self.native.Lev1Scorer.from_pretrained(
            str(self.path), device=device, dtype="bfloat16",
            chat=True, temperature=None, yes_bias=0.0,
            order_average=True, orders="reverse", sdpa_backend="default",
        )
        self.attention_model = self.engine.model
        if self.attention_model.config._attn_implementation != "sdpa":
            raise RuntimeError("Lev1 native packed block-mask SDPA is not active")
        capacity = self.engine.max_context
        limit = context_limit or 32768
        if limit > capacity:
            raise ValueError("Lev1 context limit exceeds checkpoint capacity")
        self.engine.max_context = limit
        self.settings.update({
            "dtype": "bfloat16", "attention_implementation": "sdpa",
            "max_input_tokens": limit, "checkpoint_position_limit": capacity,
            "input_length_policy": "native-full-prefix-plus-longest-block-rejection",
            "packed_max_tokens": self.engine.packed_max_tokens,
            "chunk_tokens": self.engine.chunk_tokens, "pad_multiple": self.engine.pad_multiple,
            "order_average": True, "orders": "reverse", "temperature": 1.0, "yes_bias": 0.0,
            "weights_origin": "pinned-merged-release",
            "readout": "native-label-head-packed-or-cached-block-softmax-order-average",
            "typed_mapping": "all-primitives-choice-preserving-authored-noul-and-numeric-score",
            "renderer": "native-structured-anonymous-numeric-score-v1",
        })

    def predict(self, case):
        questions = sifr_questions(case)
        predictions = []
        for question in case.questions:
            response = self.engine.score(case.state, {"decision": questions[question.id]})
            if set(response) != {"decision"}:
                raise ValueError("Lev1 returned incorrect question count")
            probabilities = response["decision"]
            keys = ([option.id for option in question.options] if question.task == "noul"
                    else [f"option_{i}" for i in range(len(question.options))])
            check_probabilities(probabilities, keys)
            aligned = dict(zip(
                [option.id for option in question.options],
                [probabilities[key] for key in keys], strict=True,
            ))
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=aligned,
            ))
        return predictions

    def close(self):
        for name, enabled in getattr(self, "_sdpa_flags", {}).items():
            flag = self.native.SDPA_FLAGS[name]
            getattr(self.torch.backends.cuda, f"enable_{flag}_sdp")(enabled)
        super().close()
