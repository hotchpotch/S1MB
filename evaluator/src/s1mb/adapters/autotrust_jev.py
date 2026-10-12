"""AutoTrust's released bare-v1 FP32 slot-head inference, without decoding."""

import hashlib
import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, state_text


def autotrust_choice_text(state, instruction, criteria):
    """Retain authored definitions and numeric levels in the native Choice template."""
    if not 2 <= len(criteria) <= 16:
        raise ValueError("AutoTrust JEV native Choice head supports 2–16 candidates")
    def render(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    lines = [f"{'ABCDEFGHIJKLMNOP'[i]}) {render(value)}"
             for i, value in enumerate(criteria.values())]
    return (f"[kind] choice\n[state] {state_text(state)}\n[question] {render(instruction)}"
            "\n[options]\n" + "\n".join(lines) + "\n[decision]:")


class AutoTrustJevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("autotrust-jev", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        safetensors = importlib.import_module("safetensors.torch")
        config = json.loads((self.path / "judge_config.json").read_text())
        if (config["slots"]["template_version"] != "bare-v1"
                or config["slots"]["ranges"]["choice"] != [8, 24]
                or config["weights_mode"] != "unmerged"):
            raise ValueError("Unsupported AutoTrust JEV slot-head configuration")
        capacity = transformers.AutoConfig.from_pretrained(str(self.path)).get_text_config().max_position_embeddings
        self.limit = 32768 if context_limit is None else context_limit
        if not 1 <= self.limit <= capacity:
            raise ValueError("AutoTrust JEV context limit exceeds checkpoint capacity")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        base = transformers.AutoModelForCausalLM.from_pretrained(
            str(self.path), dtype=self.torch.bfloat16, device_map=device,
            attn_implementation="sdpa",
        )
        self.engine = peft.PeftModel.from_pretrained(
            base, str(self.path / config["adapter_subfolder"]),
        ).merge_and_unload().eval()
        self.attention_model = self.engine
        head = safetensors.load_file(str(self.path / "head.safetensors"))
        self.weight = head["proj.weight"].to(device=device, dtype=self.torch.float32)
        self.bias = head["proj.bias"].to(device=device, dtype=self.torch.float32)
        if self.weight.shape != (24, config["hidden_size"]) or self.bias.shape != (24,):
            raise ValueError("AutoTrust JEV head shape differs from its released slots")
        # Include the independently loaded head in the common parameter census.
        self.engine.register_parameter(
            "s1mb_slot_weight", self.torch.nn.Parameter(self.weight, requires_grad=False),
        )
        self.engine.register_parameter(
            "s1mb_slot_bias", self.torch.nn.Parameter(self.bias, requires_grad=False),
        )
        self.temperature = json.loads((self.path / "calibration.json").read_text())["per_kind"]["choice"]
        if not 0 < self.temperature < float("inf"):
            raise ValueError("Invalid AutoTrust JEV calibration temperature")
        self.settings.update({
            "dtype": "bfloat16", "head_dtype": "float32", "attention_implementation": "sdpa",
            "max_input_tokens": self.limit, "checkpoint_position_limit": capacity,
            "input_length_policy": "reject-overflow", "temperature": self.temperature,
            "candidate_limit": 16, "weights_mode": "native-bfloat16-lora-merge",
            "renderer": "bare-v1-choice-authored-noul-numeric-score",
            "native_model_card_sha256": hashlib.sha256((self.path / "README.md").read_bytes()).hexdigest(),
        })

    def predict(self, case):
        wire = sifr_questions(case)
        predictions = []
        for question in case.questions:
            spec = wire[question.id]
            text = autotrust_choice_text(case.state, spec["instructions"], spec["criteria"])
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[1] > self.limit:
                raise ValueError("Complete AutoTrust JEV input exceeds context limit")
            inputs = inputs.to(self.device)
            with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                hidden = self.engine.model(**inputs).last_hidden_state[0, -1].float()
            logits = (self.weight @ hidden + self.bias) / self.temperature
            values = self.torch.softmax(logits[8:8 + len(question.options)], 0).tolist()
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions

    def close(self):
        del self.weight, self.bias
        super().close()
