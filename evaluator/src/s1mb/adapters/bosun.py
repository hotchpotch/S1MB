"""Bosun stable-slot decisions using its pinned native prompt and readout."""

import importlib
import json

from s1mb.data import Prediction

from .upstream import UpstreamAdapter


def bosun_candidates(question):
    return [
        {
            "id": f"option_{i}",
            "label": o.id
            if question.task == "noul"
            else str(o.value)
            if question.task == "score"
            else f"option_{i}",
            "description": o.description_json or o.description,
        }
        for i, o in enumerate(question.options)
    ]


class BosunAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("bosun", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        self.engine = (
            transformers.AutoModelForCausalLM.from_pretrained(
                str(self.path),
                trust_remote_code=True,
                dtype=self.torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        self.native = importlib.import_module(type(self.engine).__module__)
        self.attention_model = self.engine.model
        self.attention_model.set_attn_implementation("sdpa")
        self.context_limit = 32768
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": self.context_limit,
            "case_batch_size": 1,
            "seed": 0,
            "native_row_id": "0",
            "base_model": self.engine.config.base_model_name_or_path,
            "base_revision": self.engine.config.base_model_revision,
            "renderer": "native-stable-slots-anonymous-candidates-v1",
        }

    def predict(self, case):
        predictions = []
        for q in case.questions:
            instruction = (
                json.dumps(json.loads(q.instructions_json), ensure_ascii=False)
                if q.instructions_json is not None
                else q.instructions
            )
            instruction = "\n\n".join(x for x in (q.system_prompt, instruction) if x)
            kwargs = {
                "state": case.state,
                "instructions": instruction,
                "candidates": bosun_candidates(q),
                "decision_type": q.task,
                "seed": 0,
                "row_id": "0",
            }
            content, _, _ = self.native.render_decision_prompt(
                **kwargs,
                decision_tokens=self.engine.decision_tokens,
                prompt_schema=self.engine.config.prompt_schema,
            )
            prompt = self.engine.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.native._SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.engine.config.decision_prompt_enable_thinking,
            )
            if (
                len(self.engine.tokenizer(prompt, truncation=False)["input_ids"])
                > self.context_limit
            ):
                raise ValueError("Bosun input exceeds context limit; refusing truncation")
            result = self.engine.predict(**kwargs)
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={
                        o.id: p for o, p in zip(q.options, result["probabilities"], strict=True)
                    },
                )
            )
        return predictions
