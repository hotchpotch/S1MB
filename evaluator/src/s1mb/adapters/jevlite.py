"""JevLite native calibrated answer logits with strict full-input checks."""

import importlib
import string
from itertools import product

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import decision_row
from .upstream import UpstreamAdapter


def extended_labels(tokenizer, native_labels, count):
    """Extend the native prefix with unique single-token, leading-space labels."""
    labels = list(native_labels)
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(row) != 1 for row in ids) or len({row[0] for row in ids}) != len(ids):
        raise ValueError("Native labels must have distinct single-token encodings")
    seen = {row[0] for row in ids}
    for pair in product(string.ascii_uppercase, repeat=2):
        if len(labels) >= count:
            break
        label = " " + "".join(pair)
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) == 1 and tokens[0] not in seen:
            labels.append(label)
            seen.add(tokens[0])
    if len(labels) < count:
        raise ValueError("Tokenizer has insufficient single-token answer codes")
    return labels[:count]


class JevLiteAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("jevlite", model, revision, source, device)
        self.native = importlib.import_module("jevlite.model")
        self.prompt = importlib.import_module("jevlite.prompt")
        self.context_limit = context_limit or self.native.DEFAULT_MAX_TOKENS
        self.engine = self.native.SystemOne(
            str(self.path),
            device=device,
            dtype=self.torch.bfloat16,
            max_tokens=self.context_limit,
            batch_size=1,
        )
        self.native_label_count = len(self.prompt.LETTER_LABELS)
        count = max_candidates or self.native_label_count
        if not 2 <= count <= 255:
            raise ValueError("max_candidates must be in 2..255")
        labels = extended_labels(self.engine.tokenizer, self.prompt.LETTER_LABELS, count)
        self.prompt.LETTER_LABELS[:] = labels
        self.engine.letter_ids, self.engine.noul_ids = self.native.label_token_ids(
            self.engine.tokenizer
        )
        self.attention_model = self.engine.model
        self.settings = {
            "dtype": "bfloat16",
            "max_input_tokens": self.context_limit,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "temperature": self.engine.temperature,
            "max_candidates": count,
            "native_max_candidates": self.native_label_count,
            "answer_labels": labels,
            "capacity_extension": count > self.native_label_count,
            "renderer": "native-anonymous-choice-numeric-score-v1",
        }
        self.set_attention("sdpa")
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for q in case.questions:
            question = decision_row(case.state, q)["question"]
            self.prompt.labels_for(question)
            text = self.prompt.render(case.state, question)
            ids = self.engine.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > self.context_limit:
                raise ValueError("JevLite input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                values, _ = self.engine.distributions([(case.state, question, None)])
            ids = ["true", "false"] if q.task == "noul" else [o.id for o in q.options]
            probabilities = dict(zip(ids, values[0], strict=True))
            check_probabilities(probabilities, [o.id for o in q.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities=probabilities,
                )
            )
        return predictions
