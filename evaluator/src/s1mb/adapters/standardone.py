"""StandardOne's native wording, tokenizer boundary and per-task temperatures."""

import importlib
import sys

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import decision_row
from .jevlite import extended_labels
from .upstream import UpstreamAdapter


class StandardOneAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None, max_candidates=None):
        self.setup("standardone", model, revision, source, device)
        sys.path.insert(0, str(self.path / "server"))
        self.native = importlib.import_module("jev_adapter.protocol")
        boundary = importlib.import_module("jev_adapter.native_tokenizer")
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        if self.tokenizer is None:
            raise ValueError("Tokenizer could not be loaded")
        self.boundary = boundary.NativeTokenizer(self.tokenizer)
        self.codes = extended_labels(
            self.tokenizer, list(self.native._NATIVE_DISPLAY), max_candidates or 26
        )
        self.native.__dict__["_NATIVE_DISPLAY"] = self.codes
        self.code_ids = [
            self.tokenizer.encode(code, add_special_tokens=False)[0] for code in self.codes
        ]
        self.engine = transformers.Mistral3ForConditionalGeneration.from_pretrained(
            str(self.path),
            dtype=self.torch.bfloat16,
            device_map={"": device},
            attn_implementation="sdpa",
        ).eval()
        self.attention_model = self.engine
        self.context_limit = context_limit or 8192
        self.temperature = {"choice": 0.85, "noul": 0.85, "score": 0.70}
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit,
            "checkpoint_runtime_limit": 8192,
            "max_candidates": len(self.codes),
            "native_max_candidates": 26,
            "answer_codes": self.codes,
            "temperature": self.temperature,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "native-no-system-anonymous-choice-numeric-score-v1",
            "permutations": 1,
        }

    def predict(self, case):
        predictions = []
        for question in case.questions:
            row = decision_row(case.state, question)
            definition, count = row["question"], len(question.options)
            criteria = definition["criteria"]
            keys = (
                ["true", "false"]
                if question.task == "noul"
                else list(criteria)
                if isinstance(criteria, dict)
                else [str(i) for i in range(count)]
            )
            descriptions = (
                [criteria[key] for key in keys] if isinstance(criteria, dict) else criteria
            )
            plan = self.native.QuestionPlan(
                question.task, definition["instructions"], tuple(keys), tuple(descriptions)
            )
            if count > len(self.codes):
                raise ValueError("StandardOne candidate capacity exceeded")
            prompt = self.native.build_prompt(
                row["state"], plan, self.codes[:count], tuple(range(count)), wording="native"
            )
            ids = self.boundary.prepare(
                prompt, tuple(self.codes[:count]), tuple(self.code_ids[:count]), None, None
            )
            if len(ids) > self.context_limit:
                raise ValueError("StandardOne input exceeds context limit; refusing truncation")
            with self.torch.inference_mode():
                logits = (
                    self.engine(
                        input_ids=self.torch.tensor([ids], device=self.device),
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    .logits[0, -1]
                    .float()
                )
                values = (
                    (logits[self.code_ids[:count]] / self.temperature[question.task])
                    .softmax(-1)
                    .cpu()
                    .tolist()
                )
            raw = dict(zip(keys, values, strict=True))
            output_keys = [o.id for o in question.options] if question.task == "noul" else keys
            probabilities = {
                o.id: raw[key] for o, key in zip(question.options, output_keys, strict=True)
            }
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
