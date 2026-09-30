"""OpenThai typed slot decisions with explicit single-order inference."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


class OpenThaiAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device):
        self.setup("openthai", model, revision, source, device)
        native = importlib.import_module("openthai_systemone.client")
        self.types = importlib.import_module("openthai_systemone.types")
        self.engine = native.SystemOneClient(
            str(self.path),
            device="cuda",
            dtype=self.torch.bfloat16,
            max_total_tokens=32768,
            max_state_tokens=32768,
        )
        self.attention_model = self.engine.model.model
        self.settings = {
            "dtype": "bfloat16",
            "input_length_policy": "reject-overflow",
            "max_input_tokens": 32768,
            "case_batch_size": 1,
            "permutations": 1,
            "renderer": "native-independent-single-order-anonymous-choice-v1",
            "probability_condition": "native normalization over candidates excluding abstain",
        }
        self.set_attention("sdpa")
        self.enable_kernels()

    def predict(self, case):
        answers = {}
        for q in case.questions:
            request = questions_for_api([q], anonymous_choice=True)
            parsed = {key: self.types.parse_question(value) for key, value in request.items()}
            encoded = self.engine.fmt.encode(case.state, parsed)
            if encoded.truncated_state or encoded.n_tokens > self.settings["max_input_tokens"]:
                raise ValueError("OpenThai input exceeds context limit; refusing truncation")
            result = self.engine.system_one(case.state, request, permutations=1)
            answers.update({key: value.model_dump() for key, value in result.answers.items()})
        return decode_answers(case, answers, anonymous_choice=True)
