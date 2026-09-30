"""Kotoba's trained DeBERTa decision head and native span-pooling interface."""

import importlib

from s1mb.data import Prediction

from .deberta_attention import enable_query_tiling
from .upstream import UpstreamAdapter, state_text


class KotobaAdapter(UpstreamAdapter):
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("kotoba", model, revision, source, device)
        native = importlib.import_module("typed_decisions.open_jev")
        self.schema = importlib.import_module("typed_decisions.schema")
        self.engine = native.OpenJev.from_pretrained(str(self.path), device=device)
        self.attention_model = self.engine.model.backbone
        original_limits = [self.engine.collator.max_state, self.engine.collator.max_len]
        if context_limit is not None:
            if not 1 <= context_limit <= 32768:
                raise ValueError("Kotoba context limit must be within 1..32768")
            self.engine.collator.max_state = context_limit
            self.engine.collator.max_len = context_limit
            enable_query_tiling(self.attention_model)
        self.settings = {
            "dtype": "float32",
            "attention_implementation": "query-tiled-eager"
            if context_limit is not None
            else "native-eager",
            "input_length_policy": "reject-overflow",
            "case_batch_size": 1,
            "max_state_tokens": self.engine.collator.max_state,
            "max_input_tokens": self.engine.collator.max_len,
            "temperature": self.engine.model.temperature,
            "renderer": "native-span-enum-authored-noul-and-score-v1",
            "checkpoint_runtime_limits": original_limits,
            "position_extrapolation": (context_limit or original_limits[1]) > 512,
            "query_block_size": 128 if context_limit is not None else None,
        }

    def predict(self, case):
        state = state_text(case.state)
        collator = self.engine.collator
        if len(collator._ids(state)) > collator.max_state:
            raise ValueError("Kotoba state exceeds native limit; refusing truncation")
        predictions = []
        for q in case.questions:
            instruction = "\n\n".join(
                x for x in (q.system_prompt, q.instructions_json or q.instructions) if x
            )
            # The public convenience method discards Noul criteria. Its native
            # Question/Collator interface permits the authored descriptions.
            options = [
                (f"{o.id}: " if q.task == "noul" else f"{o.value}: " if q.task == "score" else "")
                + (o.description_json or o.description)
                for o in q.options
            ]
            question = self.schema.Question("decision", "choice", instruction, options, 0)
            batch = collator([(state, [question])], self.device)
            with self.torch.inference_mode():
                logits = self.engine.model(
                    *[
                        batch[k]
                        for k in (
                            "input_ids",
                            "attention_mask",
                            "opt_pos",
                            "opt_mask",
                            "q_pos",
                            "seg",
                        )
                    ]
                ).float()
                probabilities = (logits / self.engine.model.temperature).softmax(-1)
                raw = probabilities[0, 0, : len(q.options)].cpu().tolist()
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    question_id=q.id,
                    probabilities={o.id: p for o, p in zip(q.options, raw, strict=True)},
                )
            )
        return predictions
