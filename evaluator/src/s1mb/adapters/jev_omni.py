"""Pinned Jev-Omni native text backbone and 256-way calibrated decision head."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, state_text


def omni_inputs(state, question):
    item = questions_for_api([question], structured=True, anonymous_choice=True)[question.id]
    descriptions = item["criteria"] if question.task == "score" else list(item["criteria"].values())
    options = []
    for index, (option, description) in enumerate(zip(question.options, descriptions, strict=True)):
        label = (
            option.id
            if question.task == "noul"
            else str(option.value)
            if question.task == "score"
            else f"option_{index}"
        )
        # Native outputs use descriptions as dictionary keys; anonymous prefixes
        # distinguish duplicate descriptions without exposing dataset identifiers.
        options.append(f"{index + 1}. {label}: {state_text(description)}")
    return state_text(state), state_text(item["instructions"]), options


class JevOmniAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("jev-omni", model, revision, source, device)
        self.native = importlib.import_module("jev_omni")
        transformers = importlib.import_module("transformers")
        config = transformers.AutoConfig.from_pretrained(str(self.path))
        backbone = (
            getattr(transformers, config.architectures[0])
            .from_pretrained(
                str(self.path),
                dtype=self.torch.bfloat16,
                device_map=device,
                attn_implementation="sdpa",
            )
            .eval()
        )
        decision = json.loads((self.path / "decision_config.json").read_text())
        head = self.native._Head256(decision["hidden_size"]).to(device).eval()
        head.load_state_dict(
            self.torch.load(
                self.path / "head.pt",
                map_location=device,
                weights_only=True,
            )
        )
        _, decoder = self.native._find_backbone(backbone)
        processor = transformers.AutoProcessor.from_pretrained(str(self.path))
        self.engine = self.native.JevOmni(backbone, head, processor, decoder, device)
        self.attention_model = backbone
        self.context_limit = context_limit or 8192
        self.settings = {
            "dtype": "bfloat16-backbone-float32-head",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit,
            "max_candidates": 256,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "renderer": "native-anonymous-unique-descriptions-numeric-score-v1",
        }

    def predict(self, case):
        predictions = []
        for q in case.questions:
            if len(q.options) > 256:
                raise ValueError("Jev-Omni exceeds its 256-option head capacity")
            state, instructions, options = omni_inputs(case.state, q)
            text = self.native._prompt(state, instructions, options)
            inputs = self.engine.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": text}]}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            if inputs["input_ids"].shape[1] > self.context_limit:
                raise ValueError("Jev-Omni input exceeds context limit; refusing truncation")
            result = self.engine.predict(state=state, question=instructions, options=options)
            if set(result["probabilities"]) != set(options):
                raise ValueError("Jev-Omni returned unexpected option keys")
            probabilities = {
                option.id: result["probabilities"][key]
                for option, key in zip(q.options, options, strict=True)
            }
            check_probabilities(probabilities, [option.id for option in q.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities=probabilities,
                )
            )
        return predictions
