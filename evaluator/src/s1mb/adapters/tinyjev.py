"""TinyJev's native pointer readout with bounded independent questions."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


class TinyJevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("tinyjev", model, revision, source, device)
        native = importlib.import_module("tinyjev")
        self.engine = native.load(str(self.path), backend="torch", device=device)
        self.attention_model = self.engine.backbone.model
        original_limits = [self.engine.family.max_state, self.engine.family.max_branch]
        if context_limit is not None:
            maximum = self.attention_model.config.max_position_embeddings
            if not 1 <= context_limit <= maximum:
                raise ValueError(f"TinyJev context limit must be within 1..{maximum}")
            self.engine.family.max_state = context_limit
            self.engine.family.max_branch = context_limit
        self.settings = {
            "dtype": "float16",
            "attention_implementation": "sdpa",
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "max_state_tokens": self.engine.family.max_state,
            "max_row_tokens": self.engine.family.max_branch,
            "temperature": self.engine.family.temperature,
            "renderer": "native-independent-structured-anonymous-choice-v1",
            "checkpoint_runtime_limits": original_limits,
        }

    def predict(self, case):
        answers = {}
        for q in case.questions:
            request = questions_for_api([q], structured=True, anonymous_choice=True)
            result = self.engine.systemone({"state": case.state, "questions": request})
            answers.update(result["answers"])
        return decode_answers(case, answers, anonymous_choice=True, rounded=True)

    def metadata(self):
        info = super().metadata()
        # Native pointer parameters are NumPy arrays, outside the Torch module tree.
        head_params = sum(weight.size for weight in self.engine.family.w.values())
        if info.total_params is None or info.active_params is None:
            raise ValueError("TinyJev backbone parameter counts are unavailable")
        return info.model_copy(
            update={
                "total_params": info.total_params + head_params,
                "active_params": info.active_params + head_params,
                "settings": {**info.settings, "numpy_head_params": head_params},
            }
        )
