"""Naluri encoders: native calibrated typed decisions, one forward pass per question."""

import importlib

from s1mb.data import InferenceCase, Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches


def full_sequence(tokenizer, state, question, s1, context_limit=None):
    """Naluri's native layout (s1.build_sequence) without shortening any field.

    <s> {type} question: {instructions} </s> <mask> opt0 <mask> opt1 ... </s> {state} </s>
    By default the trained budgets apply (512 tokens in all, 192 for the question and options, 48 per
    option). An explicit context_limit replaces the total and lifts the question and option budgets.
    """
    mask = tokenizer.mask_token
    texts = [
        f"{question['type']} question: {question['instructions']}",
        *[" " + text.replace(mask, " ") for text in s1.render_options(question)],
        s1.serialize_state(state).replace(mask, " "),
    ]
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    head, options, body = encoded[0], encoded[1:-1], encoded[-1]
    if context_limit is None and any(len(option) > s1.OPT_MAX for option in options):
        raise ValueError(f"Naluri option exceeds {s1.OPT_MAX} tokens; refusing truncation")
    if context_limit is None and len(head) + sum(len(option) + 1 for option in options) > s1.HEAD_MAX_LEN:
        raise ValueError(f"Naluri question and options exceed {s1.HEAD_MAX_LEN} tokens; refusing truncation")
    ids = [tokenizer.cls_token_id, *head, tokenizer.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend([tokenizer.mask_token_id, *option])
    ids.extend([tokenizer.sep_token_id, *body, tokenizer.sep_token_id])
    limit = context_limit or s1.MAX_LEN
    if len(ids) > limit:
        raise ValueError(f"Naluri input requires {len(ids)} tokens, exceeding {limit}")
    return {"ids": ids, "markers": markers}


class NaluriAdapter(UpstreamAdapter):
    case_batch_size = 32

    def __init__(self, model: str, revision: str, source: str, device: str, context_limit=None):
        self.setup("naluri", model, revision, source, device)
        self.s1 = importlib.import_module("s1")
        self.tokenizer, self.engine = self.s1.load(str(self.path), device=device)  # FP32 weights
        self.attention_model = self.engine.encoder
        positions = self.engine.encoder.config.max_position_embeddings
        if context_limit is not None and context_limit < self.s1.MAX_LEN:
            raise ValueError(f"--context-limit must be at least the trained {self.s1.MAX_LEN} tokens")
        self.context_limit = context_limit
        self.settings = {
            "dtype": "float32",
            "autocast": False,
            "case_batch_size": self.case_batch_size,
            "microbatch_tokens": 8192,
            "renderer": "native-layout-full-fields-anonymous-choice-v1",
            "instructions": "dataset-default text (system prompt joined by a blank line)",
            "input_length_policy": "reject-overflow",
            "trained_input_budget": {"max_len": self.s1.MAX_LEN, "head_max_len": self.s1.HEAD_MAX_LEN, "option_max_len": self.s1.OPT_MAX},
            "max_len": context_limit or self.s1.MAX_LEN,
            "head_max_len": None if context_limit else self.s1.HEAD_MAX_LEN,
            "option_max_len": None if context_limit else self.s1.OPT_MAX,
            "extended_context": context_limit is not None,
            "checkpoint_context_length": positions,
            "beyond_checkpoint_context": bool(context_limit and context_limit > positions),
            "temperature": self.engine.temperature.tolist(),
            "probabilities": "native softmax over options with the checkpoint's per-type temperature",
        }

    def predict(self, case: InferenceCase) -> list[Prediction]:
        return self.predict_batch([case])[0]

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction]]:
        torch = self.torch
        rows = []
        for index, case in enumerate(cases):
            for q in case.questions:
                # Stable dataset identifiers must not become model text.
                native = questions_for_api([q], anonymous_choice=True)[q.id]
                rows.append((index, case.case_id, q, native, full_sequence(self.tokenizer, case.state, native, self.s1, self.context_limit)))
        rows.sort(key=lambda row: len(row[4]["ids"]))
        results: list[dict[str, Prediction]] = [{} for _ in cases]
        with torch.inference_mode():
            for group in candidate_batches(rows, [len(r[4]["ids"]) for r in rows], max_batch=32, token_budget=8192):
                batch = self.s1.collate([r[4] for r in group], self.tokenizer.pad_token_id)
                logits = self.engine(**{k: v.to(self.device) for k, v in batch.items()})
                for row, z in zip(group, logits, strict=True):
                    index, case_id, q, native, item = row
                    count = len(item["markers"])
                    t = self.engine.temperature[self.s1.QTYPE[native["type"]]]
                    p = torch.softmax(z[:count].float() / t, -1).cpu().tolist()
                    ids = ["false", "true"] if q.task == "noul" else [o.id for o in q.options]
                    results[index][q.id] = Prediction(
                        case_id=case_id, question_id=q.id, probabilities=dict(zip(ids, p, strict=True))
                    )
        return [[result[q.id] for q in case.questions] for result, case in zip(results, cases, strict=True)]
