"""Jeff's native prompt and automatic first-token/whole-label readout."""

import importlib
import string
from itertools import product

from s1mb.data import Prediction

from .base import questions_for_api
from .upstream import UpstreamAdapter, checkpoint_path


def anonymous_choice_codes(tokenizer, count=255):
    """Choose stable anonymous labels with distinct native leading-space tokens."""
    labels = list(string.ascii_uppercase) + [
        "".join(pair) for pair in product(string.ascii_uppercase, repeat=2)
    ]
    tokens = tokenizer([" " + label for label in labels], add_special_tokens=False)["input_ids"]
    codes, seen = [], set()
    for label, ids in zip(labels, tokens, strict=True):
        if len(ids) == 1 and ids[0] not in seen:
            codes.append(label)
            seen.add(ids[0])
            if len(codes) == count:
                return codes
    raise ValueError("Jeff tokenizer has too few distinct anonymous candidate codes")


class JeffAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("jeff", model, revision, source, device)
        self.native = importlib.import_module("jev_clf.client")
        self.schema = importlib.import_module("jev_clf.schema")
        self.readout = importlib.import_module("jev_clf.readout")
        base = "Qwen/Qwen3-4B-Instruct-2507"
        base_path, base_revision = checkpoint_path(base, "main")
        self.engine = self.native.SystemOneClient(
            base_model=str(base_path),
            adapter=str(self.path),
            device=device,
            dtype=self.torch.bfloat16,
            max_length=32768,
        )
        self.attention_model = self.engine.model
        self.choice_codes = anonymous_choice_codes(self.engine.tokenizer)
        self.set_attention("sdpa")
        self.settings.update(
            {
                "dtype": "bfloat16",
                "input_length_policy": "reject-overflow",
                "max_input_tokens": 32768,
                "case_batch_size": 1,
                "base_model": base,
                "base_revision": base_revision,
                "renderer": "native-anonymous-codebook-authored-noul-v1",
                "choice_label_codebook": self.choice_codes,
                "readout": "native-auto-first-token-or-whole-label",
                "extended_input_condition": "32768 tokens instead of runtime 2048 default",
            }
        )

    def predict(self, case):
        predictions = []
        for q in case.questions:
            item = questions_for_api([q], anonymous_choice=True)[q.id]
            if q.task == "choice":
                if len(q.options) > len(self.choice_codes):
                    raise ValueError("Jeff choice exceeds the anonymous codebook capacity")
                item["criteria"] = dict(zip(self.choice_codes, item["criteria"].values()))
            if q.task == "noul":
                question = self.schema.NoulQuestion(
                    instructions=item["instructions"],
                    criteria={"yes": item["criteria"]["true"], "no": item["criteria"]["false"]},
                )
                keys = ["yes" if o.id == "true" else "no" for o in q.options]
            else:
                cls = (
                    self.schema.ChoiceQuestion if q.task == "choice" else self.schema.ScoreQuestion
                )
                question = cls(instructions=item["instructions"], criteria=item["criteria"])
                keys = [
                    self.choice_codes[i] if q.task == "choice" else str(i)
                    for i in range(len(q.options))
                ]
            labels = self.schema.label_space(question)
            text = self.native.build_inputs(self.engine.tokenizer, case.state, question)
            ids = self.engine.tokenizer(text, return_tensors="pt", truncation=False).to(self.device)
            variants = self.readout.label_token_variants(self.engine.tokenizer, labels)
            width = ids["input_ids"].shape[1]
            if width + max(map(len, variants.values())) > self.settings["max_input_tokens"]:
                raise ValueError("Jeff input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                if self.readout.choose_mode(variants) == "first_token":
                    logits = (
                        self.engine.model(**ids, use_cache=False, logits_to_keep=1)
                        .logits[0, -1]
                        .float()
                    )
                    raw = (
                        logits[[variants[label][0] for label in labels]].softmax(-1).cpu().tolist()
                    )
                else:
                    totals = []
                    for label in labels:
                        seq = variants[label]
                        full = self.torch.tensor(
                            [ids["input_ids"][0].tolist() + seq], device=self.device
                        )
                        logits = (
                            self.engine.model(
                                input_ids=full, use_cache=False, logits_to_keep=len(seq) + 1
                            )
                            .logits[0]
                            .float()
                        )
                        totals.append(
                            sum(float(logits[i].log_softmax(-1)[tid]) for i, tid in enumerate(seq))
                        )
                    raw = self.torch.tensor(totals, dtype=self.torch.float32).softmax(-1).tolist()
            probabilities = dict(zip(labels, raw, strict=True))
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={
                        o.id: probabilities[k] for o, k in zip(q.options, keys, strict=True)
                    },
                )
            )
        return predictions
