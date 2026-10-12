"""OneJev's pinned native multimodal runtime for complete text-only requests."""

import importlib

from .base import decode_answers, questions_for_api
from .upstream import UpstreamAdapter


def onejev_questions(case):
    questions = questions_for_api(case.questions, structured=True, anonymous_choice=True)
    for question in case.questions:
        if question.task == "score":
            item = questions[question.id]
            item["criteria"] = [
                {"value": option.value, "description": description}
                for option, description in zip(question.options, item["criteria"], strict=True)
            ]
    return questions


class OneJevAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("OneJev context_limit must be positive")
        self.setup("onejev", model, revision, source, device)
        native = importlib.import_module("qev.mm_engine")
        self.schema = importlib.import_module("qev.schema")
        calibration_module = importlib.import_module("qev.calibrate")
        calibration_path = self.path / "calibration.json"
        calibration = (
            calibration_module.Calibration.load(calibration_path)
            if calibration_path.exists()
            else calibration_module.Calibration()
        )
        self.engine = native.MMDecisionEngine(
            str(self.path),
            device=device,
            dtype="bfloat16",
            head_dtype="float32",
            calibration=calibration,
            fork_mode="sequential",
            max_branch_tokens=context_limit or 32768,
            max_request_tokens=context_limit or 32768,
            cuda_graphs=False,
            gpu_preprocess=False,
            allow_local_paths=False,
        )
        self.attention_model = self.engine.model
        self.set_attention("sdpa")
        self.settings.update(
            {
                "dtype": "bfloat16-float32-head",
                "max_input_tokens": context_limit or 32768,
                "input_length_policy": "reject-overflow-native",
                "fork_mode": "sequential",
                "debias": 1,
                "questions_per_call": 1,
                "media": "none-text-benchmarks",
                "native_prompt_version": self.engine.style.name,
                "calibration_temperatures": calibration.temperatures,
                "calibration_policy": "release-file" if calibration_path.exists() else "native-T1",
                "renderer": "native-structured-anonymous-numeric-score-v1",
                "output_rounding": "native-six-decimal-bounded-renormalization",
            }
        )

    def predict(self, case):
        answers = {}
        for key, question in onejev_questions(case).items():
            request = self.schema.SystemOneRequest.model_validate(
                {"state": case.state, "questions": {"decision": question}}
            )
            response, _ = self.engine.decide(request, debias=1, media=[])
            native = response.model_dump(mode="json")["answers"]
            if set(native) != {"decision"}:
                raise ValueError("OneJev returned incorrect answer count")
            answers[key] = native["decision"]
        return decode_answers(
            case, answers, anonymous_choice=True, rounded=True, rounding_decimals=6
        )
