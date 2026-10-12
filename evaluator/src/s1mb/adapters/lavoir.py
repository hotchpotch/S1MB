"""Slot-free Lavoir decisions with general calibration and lossless input layout."""

import importlib
import importlib.metadata

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .laya import full_sequence
from .upstream import UpstreamAdapter, state_text


def lavoir_question(question):
    item = questions_for_api([question], structured=True)[question.id]
    if question.task == "choice":
        item["criteria"] = {
            f"option_{i}": value for i, value in enumerate(item["criteria"].values())
        }
    elif question.task == "score":
        item["criteria"] = [
            f"{option.value}: {state_text(description)}"
            for option, description in zip(question.options, item["criteria"], strict=True)
        ]
    return item


class LavoirAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and not 1 <= context_limit <= 1024:
            raise ValueError("Lavoir context_limit must be between 1 and 1024")
        self.setup("lavoir", model, revision, source, device)
        self.native = importlib.import_module("lavoir")
        self.engine = self.native.Lavoir.from_pretrained(str(self.path), device=device)
        self.common = importlib.import_module("laya.common")
        self.items = importlib.import_module("lavoir.items")
        self.collate = importlib.import_module("lavoir.collate")
        self.checkpoint = importlib.import_module("lavoir.checkpoint")
        self.context_limit = min(self.engine.max_len, context_limit or 1024)
        self.attention_model = self.engine.model.encoder
        self.set_attention("sdpa")
        self.torch.backends.mha.set_fastpath_enabled(False)
        self.settings.update(
            {
                "dtype": "float32-weights-bfloat16-autocast",
                "max_input_tokens": self.context_limit,
                "head_max_len": 256,
                "input_length_policy": "reject-overflow",
                "calibration": "native-slot-free-general",
                "temperature_general": self.engine.config.get("temperature_general"),
                "slots": 0,
                "laya_version": importlib.metadata.version("laya"),
                "renderer": "native-slot-free-full-fields-anonymous-numeric-score-v1",
            }
        )

    def predict(self, case):
        predictions = []
        for question in case.questions:
            internal = self.items.to_internal(lavoir_question(question))
            item = full_sequence(
                self.engine.tokenizer, case.state, internal, self.common, self.context_limit, 256
            )
            ids, markers = item["ids"], item["markers"]
            segments = [0] * len(ids)
            end = ids.index(self.engine.tokenizer.sep_token_id, markers[-1])
            segments[markers[0] : end] = [1] * (end - markers[0])
            batch = self.collate.collate_voi(
                [
                    {
                        "ids": ids,
                        "option_pos": markers,
                        "slot_pos": [],
                        "seg_ids": segments,
                        "qtype": item["qtype"],
                    }
                ],
                self.engine.tokenizer.pad_token_id,
            )
            inputs = self.collate.model_inputs(batch, self.device)
            inputs["slot_pos"] = None
            with (
                self.torch.inference_mode(),
                self.torch.autocast("cuda", dtype=self.torch.bfloat16),
            ):
                result = self.engine.model(**inputs)
            temperatures = self.checkpoint.calibration(self.engine.config, None, False)
            temperature = self.checkpoint.temperature_tensor(self.engine.model, temperatures)[
                item["qtype"]
            ]
            raw = (
                (result["logits"][0, : len(markers)].float() / temperature.clamp_min(1e-3))
                .softmax(-1)
                .cpu()
                .tolist()
            )
            keys = (
                ["false", "true"] if question.task == "noul" else [o.id for o in question.options]
            )
            probabilities = dict(zip(keys, raw, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
