"""Manchego's short and extended native candidate-code prompts."""

import importlib
import json

from s1mb.data import Prediction

from .upstream import UpstreamAdapter, state_text


def manchego_prompt(native, tokenizer, state, question):
    instruction = (
        json.loads(question.instructions_json)
        if question.instructions_json is not None
        else question.instructions
    )
    if question.system_prompt:
        instruction = question.system_prompt + "\n\n" + state_text(instruction)
    options = [
        (
            o.id
            if question.task == "noul"
            else str(o.value)
            if question.task == "score"
            else f"option_{i}",
            json.loads(o.description_json) if o.description_json is not None else o.description,
        )
        for i, o in enumerate(question.options)
    ]
    codes = native.codebook(len(options))
    if len(options) > 26:
        prompt = native.render(tokenizer, state, question.task, instruction, options, codes)
    else:
        texts = native.option_texts(question.task, options)
        menu = "\n".join(f"{code} = {text}" for code, text in zip(codes, texts, strict=True))
        user = (
            f"{state_text(instruction)}\n\nState:\n{state_text(state)}\n\nOptions:\n{menu}"
            "\n\nReply with only the letter of the best option."
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt, codes


class ManchegoAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("manchego", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        self.native = importlib.import_module("contract_v2")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        self.engine = (
            transformers.AutoModelForCausalLM.from_pretrained(
                str(self.path), dtype=self.torch.bfloat16, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        self.attention_model = self.engine
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": 32768,
            "case_batch_size": 1,
            "temperature": 1.0,
            "renderer": "native-short-or-contract-v2-anonymous-choice-v1",
            "extended_input_condition": "32768 admission; trained to approximately 9000 tokens",
        }
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for q in case.questions:
            prompt, codes = manchego_prompt(self.native, self.tokenizer, case.state, q)
            ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if ids["input_ids"].shape[1] > self.settings["max_input_tokens"]:
                raise ValueError("Manchego input exceeds context limit; refusing truncation")
            code_ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in codes]
            if any(len(c) != 1 for c in code_ids):
                raise ValueError("Manchego candidate code is not a single token")
            with self.torch.inference_mode():
                logits = (
                    self.engine(**ids.to(self.device), use_cache=False, logits_to_keep=1)
                    .logits[0, -1]
                    .float()
                )
                probabilities = logits[[c[0] for c in code_ids]].softmax(-1).cpu().tolist()
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: p for o, p in zip(q.options, probabilities, strict=True)},
                )
            )
        return predictions
