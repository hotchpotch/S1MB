"""Openjev's released shim with a local, exact candidate-logit transport."""

import importlib
from types import SimpleNamespace
from typing import Any, cast

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter, decoder_device_map, state_text


class OpenjevShimAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("openjev-shim", model, revision, source, device)
        transformers = cast(Any, importlib.import_module("transformers"))
        self.native = cast(Any, importlib.import_module("shim"))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        count = self.torch.cuda.device_count()
        placement = decoder_device_map(self.path, device, self.torch, transformers)
        self.engine = transformers.AutoModelForCausalLM.from_pretrained(
            str(self.path),
            dtype=self.torch.bfloat16,
            attn_implementation="sdpa",
            device_map=placement,
        ).eval()
        if any(p.device.type != "cuda" for p in self.engine.parameters()):
            raise RuntimeError("Openjev must fit entirely on the explicitly visible GPUs")
        self.attention_model = self.engine
        self.native.TEMP = 0.85
        self.native.NOUL_T = 1.829074
        self.native.NOUL_BIAS = 0.0
        self.native.TARGETED = True
        self.native.PERMS = 1
        self.native.PAD = 0
        self.native.LAYOUT = ""
        self.native._ids.clear()
        for letter in self.native.LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("Openjev candidate letter is not a single token")
            self.native._ids[letter] = ids[0]
        self.native.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self.complete))
        )
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": 32768,
            "case_batch_size": 1,
            "gpu_count": count,
            "device_map": placement,
            "temperature": 0.85,
            "noul_temperature": 1.829074,
            "noul_bias": 0.0,
            "renderer": "native-shim-pyrepr-anonymous-choice-v1",
            "readout": "exact-candidate-logits-native-large-menu-composition",
        }
        self.enable_kernels()

    def complete(self, *, messages, extra_body, **kwargs):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        if ids["input_ids"].shape[1] > self.settings["max_input_tokens"]:
            raise ValueError("Openjev input exceeds context limit; refusing truncation")
        allowed = extra_body["logprob_token_ids"]
        with self.torch.inference_mode():
            logits = self.engine(**ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            values = logits[allowed].log_softmax(-1).cpu().tolist()
        content = SimpleNamespace(
            top_logprobs=[
                SimpleNamespace(token=f"token_id:{i}", logprob=p)
                for i, p in zip(allowed, values, strict=True)
            ]
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[content]))],
            usage=SimpleNamespace(prompt_tokens=ids["input_ids"].shape[1]),
        )

    def predict(self, case):
        answers = {}
        for key, question in questions_for_api(
            case.questions, structured=True, anonymous_choice=True
        ).items():
            # The checkpoint's training contract used Python repr for structured instructions.
            if not isinstance(question["instructions"], str):
                question["instructions"] = str(question["instructions"])
            answers[key], _ = self.native.ANSWER[question["type"]](
                self.native.State(state_text(case.state)), question
            )
        return decode_answers(case, answers, rounded=True, anonymous_choice=True)
