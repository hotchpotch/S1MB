"""Logan Markewich's GLiFormer Jeff, distinct from Firelex and Gestalt Jeff."""

import importlib
from importlib.metadata import version

from s1mb.data import Prediction, check_probabilities

from .firelex_jeff import decision_row
from .opendecision import StrictTokenizer
from .upstream import UpstreamAdapter, state_text


class GliformerJeffAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("gliformer-jeff", model, revision, source, device)
        backend = importlib.import_module("jeff.backends.torch_backend")
        self.schema = importlib.import_module("jeff.core.schemas")
        self.groups = importlib.import_module("jeff.core.groups")
        self.answers = importlib.import_module("jeff.core.answers")
        self.state = importlib.import_module("jeff.core.state")
        self.engine = backend.TorchBackend(
            str(self.path),
            device=device,
            dtype="bfloat16",
            attn_kernel="flash",
            compile_model=False,
            warmup=False,
            batch_size=1,
        )
        self.options = self.groups.PromptOptions(isolate="all")
        self.context_limit = context_limit or 8192
        processor = self.engine.model.data_processor
        original_limit = processor.config.max_len
        processor.config.max_len = self.context_limit
        processor.transformer_tokenizer = StrictTokenizer(
            processor.transformer_tokenizer, self.context_limit
        )
        process = self.engine.model._process_multitask_batches

        def checked_process(*args, **kwargs):
            result = process(*args, **kwargs)
            decoded = result[0].get("classification", [])
            if len(decoded) != 1 or len(decoded[0]) != len(self.expected_groups):
                raise ValueError("GLiFormer returned missing classification groups")
            for group, predictions in zip(self.expected_groups, decoded[0], strict=True):
                if {row["class_name"] for row in predictions} != set(group.labels):
                    raise ValueError("GLiFormer returned missing or unexpected labels")
            return result

        self.engine.model._process_multitask_batches = checked_process
        self.settings = {
            "dtype": "bfloat16",
            "gliformer_version": version("gliformer"),
            "gliner_version": version("gliner"),
            "attention_implementation": self.engine.attn_kernel,
            "max_input_tokens": self.context_limit,
            "checkpoint_runtime_limit": original_limit,
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "temperature": self.answers.DEFAULT_TEMPERATURE,
            "probability_origin": "native-temperature-renormalized-independent-sigmoids",
            "output_rounding": False,
            "renderer": "native-isolated-anonymous-choice-numeric-score-v1",
        }

    def predict(self, case):
        predictions = []
        state = case.state if isinstance(case.state, (str, dict, list)) else state_text(case.state)
        text = self.state.serialize_state(state, self.options.state_format)
        tokens, _, _ = self.engine.model.prepare_inputs([text])
        if len(tokens[0]) > self.context_limit:
            raise ValueError("GLiFormer word count exceeds context limit; refusing truncation")
        for question in case.questions:
            row = decision_row(state, question)
            request = self.schema.SystemOneRequest(
                model=self.model_id,
                state=state,
                questions={"decision": row["question"]},
            )
            groups = self.groups.build_groups(request.questions, self.options)
            self.expected_groups = groups
            raw = self.engine.score([text], [groups])[0].scores["decision"]
            values = self.answers.normalize(raw, self.answers.DEFAULT_TEMPERATURE)
            ids = ["true", "false"] if question.task == "noul" else [o.id for o in question.options]
            probabilities = dict(zip(ids, values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
