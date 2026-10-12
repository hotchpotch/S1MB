"""Native Gevva margin-calibrated decisions with complete-input admission."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .sifr import sifr_questions
from .upstream import UpstreamAdapter, state_text


class GevvaAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        limit = 32768 if context_limit is None else context_limit
        if not 64 <= limit <= 131072:
            raise ValueError("Gevva context_limit must be between 64 and 131072")
        self.setup("gevva", model, revision, source, device)
        self.native = importlib.import_module("gemma4_cross_encoder")
        self.engine = self.native.GevvaCrossEncoder(
            model_name_or_path=str(self.path), device=device, dtype=self.torch.bfloat16,
            max_length=limit, batch_size=4,
        )
        self.settings.update({
            "dtype": "bfloat16", "max_input_tokens": limit,
            "input_length_policy": "reject-any-native-pair-truncation",
            "calibration": "native-nli-temperature-and-bucketed-margin-softmax",
            "renderer": "native-choice-authored-noul-numeric-score",
            "candidate_batch_size": 4,
        })

    def predict(self, case):
        predictions = []
        wire = sifr_questions(case)
        context = state_text(case.state)
        for question in case.questions:
            spec = wire[question.id]
            instruction = spec["instructions"]
            if not isinstance(instruction, str):
                instruction = json.dumps(instruction, ensure_ascii=False)
            texts = [v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                     for v in spec["criteria"].values()]
            keys = list(spec["criteria"])
            premise = f"{context.strip()}\nQuestion: {instruction.strip()}" if instruction else context.strip()
            for key, text in zip(keys, texts, strict=True):
                hypothesis = f"The correct answer is: {key}: {text}"
                bounded = self.native.tokenize_nli_pair_safe(
                    self.engine.tokenizer, premise, hypothesis, self.engine.max_length,
                )
                complete = self.native.tokenize_nli_pair_safe(
                    self.engine.tokenizer, premise, hypothesis, 1048576,
                )
                if bounded != complete:
                    raise ValueError("Complete Gevva input exceeds native pair budget")
            candidates = [f"{key}: {text}" for key, text in zip(keys, texts, strict=True)]
            result = self.engine.decide(context, instruction, candidates)
            probabilities = dict(zip([o.id for o in question.options], result.scores, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(Prediction(case_id=case.case_id, question_id=question.id,
                                          probabilities=probabilities))
        return predictions
