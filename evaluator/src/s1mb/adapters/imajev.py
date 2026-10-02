"""Imajev's CUDA decision readout, conditioned on declared benchmark options."""

import importlib
import json
import sys

from s1mb.data import Prediction, check_probabilities

from .nimble import nimble_field
from .upstream import UpstreamAdapter, checkpoint_path


class ImajevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("imajev", model, revision, source, device)
        sys.path.insert(0, str(self.source / "scripts"))
        runtime = importlib.import_module("torch_decision")
        self.native = importlib.import_module("vision_decision.scoring")
        self.schema = importlib.import_module("vision_decision.contracts")
        calibration = importlib.import_module("vision_decision.calibration")
        peft = importlib.import_module("peft")
        config = json.loads((self.path / "adapter_config.json").read_text())
        base = config["base_model_name_or_path"]
        if "/snapshots/" in base:
            repo = base.split("models--", 1)[1].split("/snapshots/", 1)[0].replace("--", "/")
            base_revision = base.rsplit("/", 1)[-1]
        else:
            repo, base_revision = base, config.get("revision") or "main"
        base_path, base_revision = checkpoint_path(repo, base_revision)
        self.context_limit = context_limit or 4096
        self.engine = runtime.TorchDecision(str(base_path), device, max_length=self.context_limit)
        self.engine.model = peft.PeftModel.from_pretrained(self.engine.model, str(self.path)).eval()
        self.engine.enable_readout(str(self.path), trainable=False)
        self.calibration = calibration.TemperatureCalibrator.load(self.path / "calibration.json")
        self.attention_model = self.engine.model
        self.settings = {
            "dtype": "bfloat16",
            "base_model": repo,
            "base_revision": base_revision,
            "max_input_tokens": self.context_limit,
            "max_candidates": self.engine.max_options,
            "input_length_policy": "reject-overflow",
            "rotations": 1,
            "calibration": self.calibration.to_dict(),
            "prompt_layout": self.engine.prompt_layout,
            "probability_origin": "calibrated-decision-logits-conditioned-on-declared-options",
            "unknown_policy": "native-prompt-retained-conditional-probabilities",
            "renderer": "native-authored-noul-anonymous-choice-numeric-score-v1",
            "field_text_policy": "preserve-full-text-subject-to-token-limit",
        }
        self.set_attention("sdpa")
        self.enable_kernels()

    def predict(self, case):
        predictions = []
        for question in case.questions:
            keys, schema = nimble_field(question)
            definition = schema["decision"]
            # S1MB validates these fields; native HTTP character limits must not
            # truncate benchmark rubrics. Token capacity is checked by prepare.
            if question.task == "noul":
                field = self.schema.BooleanField.model_construct(
                    id="decision",
                    type="boolean",
                    question=definition["description"],
                    yes_description=definition["choice_descriptions"]["true"],
                    no_description=definition["choice_descriptions"]["false"],
                )
                keys = ["true", "false"]
            else:
                options = [
                    self.schema.Option.model_construct(
                        value=key,
                        description=definition["choice_descriptions"][key],
                    )
                    for key in keys
                ]
                field = self.schema.ChoiceField.model_construct(
                    id="decision",
                    type="choice",
                    question=definition["description"],
                    options=options,
                )
            header, choices, texts = self.native.compile_question(
                field, case.state, self.engine.prompt_layout
            )
            labels = self.engine.labels(len(choices))
            prompt = header + "\n".join(
                f"{label}: {text}" for label, text in zip(labels, texts, strict=True)
            )
            _, inputs, candidate_ids = self.engine.prepare(None, prompt, labels)
            with self.torch.inference_mode():
                logits = self.engine.candidate_logits(inputs, candidate_ids).float()
                temperature = self.calibration.temperature(question.task, len(keys)) or 1.0
                # Conditioning removes the native unknown branch without changing
                # odds between the benchmark's declared options.
                values = (logits[:-1] / temperature).softmax(-1).cpu().tolist()
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
