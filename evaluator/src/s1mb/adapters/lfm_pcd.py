"""Pinned LFM2.5 PCD atomic-code decisions with explicit complete-input budgets."""

import importlib
from dataclasses import asdict

from s1mb.data import Prediction, check_probabilities

from .lfm_rlcd import rlcd_inputs
from .upstream import UpstreamAdapter


def admit_complete_input(engine, native, context, schema, context_limit):
    compiled = engine.compile(schema, "token")
    prefix = native.prompt_tokens(engine.tokenizer, compiled, context, engine.limits)
    width = max(len(field.suffix) for field in compiled.fields)
    if len(prefix) + width > context_limit:
        raise ValueError("PCD complete input exceeds context limit; refusing truncation")


def constrained_with_memory_retry(engine, context, schema):
    """Retry native admission once after releasing unused CUDA allocator blocks."""
    try:
        return engine.constrained(context, schema, mode="token")
    except ValueError as error:
        if str(error) != (
            "request exceeds conservative GPU memory budget; reduce prompt or branch batch size"
        ):
            raise
    torch = importlib.import_module("torch")
    with torch.cuda.device(engine.device):
        torch.cuda.empty_cache()
    return engine.constrained(context, schema, mode="token")


class LFMPCDAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("PCD context_limit must be positive")
        self.setup("lfm-pcd", model, revision, source, device)
        self.native = importlib.import_module("pcd.engine")
        config = importlib.import_module("pcd.config")
        self.context_limit = context_limit or 32768
        # These are explicit supported native Limits, including long authored
        # enum descriptions. No field/value text is cut to fit a default budget.
        limits = config.Limits(
            max_prompt_tokens=self.context_limit, max_value_tokens=self.context_limit,
            max_input_chars=self.context_limit * 16, max_schema_chars=self.context_limit * 16,
            branch_batch_size=1, projection_batch_size=32,
        )
        self.engine = self.native.Engine(
            model_id=str(self.path), revision=self.revision, device=device,
            dtype="float32", attention="sdpa", limits=limits, local_files_only=True,
        )
        self.attention_model = self.engine.model
        if self.context_limit > self.attention_model.config.max_position_embeddings:
            raise ValueError("PCD context limit exceeds checkpoint capacity")
        self.settings.update({
            "dtype": "float32", "attention_implementation": "sdpa",
            "mode": "token", "calibrated": False, "temperature": 1.0,
            "probability_origin": "native-uncalibrated-restricted-atomic-code-softmax",
            "max_input_tokens": self.context_limit, "limits": asdict(limits),
            "input_length_policy": "reject-complete-prefix-plus-field-suffix-before-prefill",
            "weights_origin": "bundled-release-unchanged-base-weights",
            "renderer": "native-pcd-structured-anonymous-numeric-score-v1",
            "memory_admission": "native-conservative-gpu-budget",
            "memory_retry": "release-unused-cuda-cache-on-native-admission-rejection-once",
        })

    def predict(self, case):
        predictions = []
        for question in case.questions:
            context, schema, labels = rlcd_inputs(case.state, question)
            admit_complete_input(self.engine, self.native, context, schema, self.context_limit)
            response = constrained_with_memory_retry(self.engine, context, schema)
            if response["mode"] != "token" or response["calibrated"] is not False:
                raise ValueError("PCD returned an unexpected scoring mode")
            if set(response["fields"]) != {"decision"}:
                raise ValueError("PCD returned incorrect field names")
            candidates = response["fields"]["decision"]["candidates"]
            if [candidate["value"] for candidate in candidates] != labels:
                raise ValueError("PCD returned different candidate values or order")
            probabilities = {
                option.id: float(candidate["probability"])
                for option, candidate in zip(question.options, candidates, strict=True)
            }
            check_probabilities(probabilities, [option.id for option in question.options])
            predictions.append(Prediction(
                case_id=case.case_id, question_id=question.id, probabilities=probabilities,
            ))
        return predictions
