"""Juspay Jev-one's bundled prompt, marker table and two-order readout."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import decision_row
from .upstream import UpstreamAdapter


def render_native_state(native, state):
    """Keep non-chat records intact when they happen to contain role fields."""
    messages = state.get("messages") if isinstance(state, dict) else state
    if (
        isinstance(messages, list)
        and all(isinstance(message, dict) and "role" in message for message in messages)
        and any(
            "content" not in message or not isinstance(message["role"], str) for message in messages
        )
    ):
        return json.dumps(state, indent=2)
    return native.render_state(state)


class JevOneAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("jevone", model, revision, source, device)
        self.native = importlib.import_module("server")
        transformers = importlib.import_module("transformers")
        config = transformers.AutoConfig.from_pretrained(str(self.path))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        if self.tokenizer is None:
            raise ValueError("Tokenizer could not be loaded")
        self.engine = (
            getattr(transformers, config.architectures[0])
            .from_pretrained(
                str(self.path),
                dtype=self.torch.bfloat16,
                device_map={"": device},
                attn_implementation="sdpa",
            )
            .eval()
        )
        self.attention_model = self.engine
        self.context_limit = context_limit or 32768
        for marker in self.native.MARKERS:
            if self.tokenizer.encode(" " + marker["m"], add_special_tokens=False) != [marker["id"]]:
                raise ValueError("Jev-one tokenizer does not match its released marker table")
        self.native.__dict__["label_ids"] = lambda: self.native.EXPECTED_LABEL_IDS
        self.native.__dict__["_post"] = self.forward
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit,
            "max_candidates": 255,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "temperature": self.native.TEMP,
            "option_orders": 2,
            "renderer": "native-release-markers-anonymous-choice-numeric-score-v1",
            "runtime": "transformers",
            "output_rounding": False,
            "structured_state_policy": "native-chat-or-json-for-non-chat-role-records",
        }
        self.enable_kernels()

    def forward(self, path, payload, **kwargs):
        if path != "/generate":
            raise ValueError("Unexpected native Jev-one operation")
        rows = [
            self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in payload["text"]
        ]
        if any(len(row) > self.context_limit for row in rows):
            raise ValueError("Jev-one input exceeds context limit; refusing truncation")
        results = []
        for ids, candidates in zip(rows, payload["token_ids_logprob"], strict=True):
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
                logprobs = logits.log_softmax(-1)[candidates].cpu().tolist()
            results.append(
                {
                    "meta_info": {
                        "output_token_ids_logprobs": [
                            [
                                [p, token, None]
                                for p, token in zip(logprobs, candidates, strict=True)
                            ]
                        ],
                        "prompt_tokens": len(ids),
                    }
                }
            )
        return results

    def predict(self, case):
        predictions = []
        for question in case.questions:
            if len(question.options) > 255:
                raise ValueError("Jev-one candidate capacity exceeded")
            row = decision_row(case.state, question)
            instruction, descriptions, _ = self.native.PREP[question.task](row["question"])

            # Structured values remain complete; the release's f-string prompt
            # otherwise renders Python repr rather than stable JSON.
            def text(value):
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

            values, _, _ = self.native.read_options_batch(
                render_native_state(self.native, row["state"]),
                [(text(instruction), [text(d) for d in descriptions], question.task)],
            )[0]
            keys = (
                ["true", "false"] if question.task == "noul" else [o.id for o in question.options]
            )
            probabilities = dict(zip(keys, values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
