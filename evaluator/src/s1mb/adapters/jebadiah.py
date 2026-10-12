"""Jebadiah's native AINode renderer and calibrated FP32 candidate readout."""

import importlib
import json
import math

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, state_text


def jebadiah_questions(case):
    """Preserve structured values and criterion order through native Choice rendering."""
    rendered = sifr_questions(case)
    for definition in rendered.values():
        definition["instructions"] = state_text(definition["instructions"])
        definition["criteria"] = {
            key: state_text(value) for key, value in definition["criteria"].items()
        }
    return rendered


class JebadiahAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("jebadiah", model, revision, source, device)
        self.native = importlib.import_module("jebadiah_model")
        if not self.native.FP32_CANDIDATE_LOGITS:
            raise ValueError("Jebadiah requires its native FP32 candidate readout")
        adapter_config = self.path / "adapter_config.json"
        if adapter_config.exists():
            base = json.loads(adapter_config.read_text())["base_model_name_or_path"]
            base_revision = "68c46c4b3498877f3ef123c856ecfde50c39f404"
            load_path, load_revision = base, base_revision
            checkpoint_format = "lora-adapter"
        else:
            manifest = json.loads((self.path / "jebadiah.json").read_text())
            base, base_revision = manifest["base"].rsplit("@", 1)
            load_path, load_revision = str(self.path), None
            checkpoint_format = "merged-bfloat16"
        tokenizer = self.native.load_tokenizer(load_path, load_revision)
        model_object = self.native.load_base(
            load_path,
            load_revision,
            attn_implementation="sdpa",
            dtype=self.torch.bfloat16,
            device=device,
        )
        if checkpoint_format == "lora-adapter":
            model_object = self.native.load_adapter(model_object, str(self.path))
        self.engine = self.native.Scorer(
            model_object,
            tokenizer,
            max_tokens=32768,
            temperatures=self.native.read_temperatures(str(self.path)),
            device=device,
        )
        if any(not math.isfinite(t) or t <= 0 for t in self.engine.temperatures.values()):
            raise ValueError("Invalid native Jebadiah temperature")
        self.attention_model = model_object
        self.settings = {
            "dtype": "bfloat16-backbone-float32-candidate-logits",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": 32768,
            "case_batch_size": 1,
            "base_model": base,
            "base_revision": base_revision,
            "checkpoint_format": checkpoint_format,
            "temperatures": self.engine.temperatures,
            "renderer": "native-ainode-extended-alphabet-anonymous-choice-v1",
            "extended_input_condition": "32768 tokens instead of runtime 2048 default",
            "typed_mapping": "all-choice-transport-original-task-temperature-authored-order-numeric-score",
        }
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        questions = jebadiah_questions(case)
        for q in case.questions:
            native = questions[q.id]
            rendered = self.engine.render(case.state, native)
            if rendered.truncated:
                raise ValueError("Jebadiah input exceeds context limit; refusing truncation")
            ids = self.engine.tok.encode(rendered.prompt, add_special_tokens=False)
            if len(ids) > self.engine.max_tokens:
                raise ValueError("Jebadiah input exceeds context limit; refusing truncation")
            for letter, candidate in zip(rendered.letters, rendered.cand_ids, strict=True):
                continuation = self.engine.tok.encode(rendered.prompt + letter, add_special_tokens=False)
                if continuation != [*ids, candidate]:
                    raise ValueError("Jebadiah label is not a single continuation token")
            probabilities = self.engine.score_rendered([(rendered, q.task)])[0]
            raw = dict(zip(rendered.keys, probabilities, strict=True))
            keys = [
                o.id if q.task == "noul" else f"option_{i}"
                for i, o in enumerate(q.options)
            ]
            if set(raw) != set(keys):
                raise ValueError("Jebadiah returned unexpected candidates")
            check_probabilities(raw, keys)
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: raw[k] for o, k in zip(q.options, keys, strict=True)},
                )
            )
        return predictions
