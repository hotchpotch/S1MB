"""Smalljev semantic option-span scoring without native state truncation."""

import importlib
import itertools
import json
import string

from s1mb.data import Prediction, check_probabilities

from .nimble import nimble_field
from .upstream import UpstreamAdapter, checkpoint_path, state_text


class SmallJevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("smalljev", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        heads = importlib.import_module("smalljev.heads")
        self.native = importlib.import_module("smalljev.semantic")
        config = json.loads((self.path / "semantic-v9-lora/adapter_config.json").read_text())
        base_id = config["base_model_name_or_path"]
        if base_id.startswith("/"):
            raise ValueError("Smalljev base checkpoint requires an explicit public model ID")
        base, base_revision = checkpoint_path(base_id, config.get("revision") or "main")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(base))
        base_model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                str(base),
                dtype=self.torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        self.engine = peft.PeftModel.from_pretrained(
            base_model, str(self.path / "semantic-v9-lora")
        ).eval()
        self.scorer = (
            heads.OptionScorerHead.load(self.path / "semantic-v9-scorer.pt").to(device).eval()
        )
        self.torch.load(
            self.path / "semantic-v9-noulscore.pt", map_location="cpu", weights_only=True
        )
        self.context_limit = context_limit or 2560
        self.capacity = max_candidates or 26
        if not 2 <= self.capacity <= 255:
            raise ValueError("Smalljev candidate capacity must be between 2 and 255")
        labels = list(string.ascii_uppercase) + [
            "".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)
        ]
        self.native.__dict__["LETTERS"] = labels[: self.capacity]
        self.attention_model = self.engine
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "base_model": base_id,
            "base_revision": base_revision,
            "max_input_tokens": self.context_limit,
            "checkpoint_runtime_limit": 2560,
            "max_candidates": self.capacity,
            "native_max_candidates": 26,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "native-semantic-rubric-anonymous-choice-numeric-score-v1",
        }

    def predict(self, case):
        predictions = []
        for question in case.questions:
            keys, schema = nimble_field(question)
            definition = schema["decision"]
            rubric = definition["choice_descriptions"]
            if question.task == "noul":
                keys = ["false", "true"]
                options = ["no", "yes"]
                rubric = {"no": rubric["false"], "yes": rubric["true"]}
            elif question.task == "score":
                options = [f"{key}: {rubric[key]}" for key in keys]
            else:
                options = keys
            if len(keys) > self.capacity:
                raise ValueError("Smalljev candidate capacity exceeded")
            text = (
                definition["description"]
                + "\nAllowed answers and rubric: "
                + json.dumps(rubric, ensure_ascii=False)
            )
            if question.task == "noul":
                text += "\nAnswer with yes or no."
            # An unbounded construction disables the native shortening loop;
            # reject the intact token sequence before any model forward.
            ids, spans = self.native.build_semantic_ids(
                self.tokenizer, state_text(case.state), text, options, max_len=2**63 - 1
            )
            if len(ids) > self.context_limit or any(a >= b for a, b in spans):
                raise ValueError("Smalljev input exceeds capacity or has an empty option span")
            with self.torch.inference_mode():
                hidden = (
                    self.engine(
                        input_ids=self.torch.tensor([ids], device=self.device),
                        use_cache=False,
                        output_hidden_states=True,
                        logits_to_keep=1,
                    )
                    .hidden_states[-1][0]
                    .float()
                )
                representations = self.torch.stack([hidden[a:b].mean(0) for a, b in spans])
                values = self.scorer.probs(representations.unsqueeze(0))[0].float().cpu().tolist()
            by_key = dict(zip(keys, values, strict=True))
            output_keys = [o.id for o in question.options] if question.task == "noul" else keys
            probabilities = {
                o.id: by_key[key] for o, key in zip(question.options, output_keys, strict=True)
            }
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
