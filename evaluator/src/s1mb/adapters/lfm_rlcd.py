"""Bounded native RLCD branch likelihoods mapped to uncalibrated distributions."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches, state_text


def rlcd_inputs(state, question):
    """Keep structured authored definitions and numeric levels in native enum values."""
    item = questions_for_api([question], structured=True)[question.id]
    if question.task == "noul":
        definitions = [item["criteria"][o.id] for o in question.options]
    else:
        definitions = (
            list(item["criteria"].values()) if question.task == "choice" else item["criteria"]
        )
    labels = [
        f"level {i}: {option.value}: {state_text(definition)}"
        if question.task == "score"
        else f"option {i}: {state_text(definition)}"
        for i, (option, definition) in enumerate(zip(question.options, definitions, strict=True))
    ]
    schema = {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": labels,
                "description": state_text(item["instructions"]),
            }
        },
        "required": ["decision"],
        "additionalProperties": False,
    }
    return state_text(state), schema, labels


def rlcd_branches(engine, context, schema, context_limit):
    """Check complete prefix and value paths before allocating an inference cache."""
    prefix = engine.encode(engine.prompt(context, schema))
    suffix = engine.encode('  "decision": ')
    branches = []
    for label in schema["properties"]["decision"]["enum"]:
        value = engine.encode(json.dumps(label, ensure_ascii=False) + "\n")
        branch = suffix + value
        if len(prefix) + len(branch) > context_limit:
            raise ValueError("RLCD complete input exceeds context limit; refusing truncation")
        branches.append((branch, len(suffix), value))
    return prefix, branches


class LFMRLCDAdapter(UpstreamAdapter):
    case_batch_size = 1
    candidate_batch_size = 8
    microbatch_tokens = 32768

    def __init__(self, model, revision, source, device, context_limit=None):
        if context_limit is not None and context_limit < 1:
            raise ValueError("RLCD context_limit must be positive")
        self.setup("lfm-rlcd", model, revision, source, device)
        self.native = importlib.import_module("rlcd.engine")
        transformers = importlib.import_module("transformers")
        # The native constructor downloads a second base checkpoint. Build the
        # same engine with the release's bundled, byte-identical base weights.
        self.engine = self.native.Engine.__new__(self.native.Engine)
        self.engine.device, self.engine.dtype = device, "float32"
        self.engine.tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.path))
        if self.engine.tokenizer is None:
            raise ValueError("RLCD checkpoint did not provide a tokenizer")
        self.engine.model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                str(self.path), dtype=self.torch.float32, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        self.engine.model.requires_grad_(False)
        position_limit = self.engine.model.config.max_position_embeddings
        self.context_limit = context_limit or min(position_limit, 32768)
        if self.context_limit > position_limit:
            raise ValueError("RLCD context_limit exceeds checkpoint capacity")
        if self.engine.tokenizer.pad_token_id is None:
            raise ValueError("RLCD tokenizer must define its native padding token")
        self.settings.update(
            {
                "dtype": "float32",
                "attention_implementation": "sdpa",
                "max_input_tokens": self.context_limit,
                "checkpoint_position_limit": position_limit,
                "input_length_policy": "reject-overflow",
                "candidate_batch_size": self.candidate_batch_size,
                "microbatch_tokens": self.microbatch_tokens,
                "probability_origin": "uncalibrated-softmax-native-full-value-log-likelihood",
                "score_temperature": 1.0,
                "length_normalization": False,
                "renderer": "native-schema-structured-anonymous-numeric-score-v1",
                "cache_branching": "native-deepcopy-reorder-including-convolution",
                "weights_origin": "bundled-release-base-weights",
            }
        )

    def branch_scores(self, context, schema):
        prefix, branches = rlcd_branches(self.engine, context, schema, self.context_limit)
        scores = []
        with self.torch.inference_mode():
            cache = self.engine.model(
                self.engine.tensor([prefix]), use_cache=True, logits_to_keep=1
            ).past_key_values
            for group in candidate_batches(
                branches,
                [len(prefix) + len(b[0]) for b in branches],
                max_batch=self.candidate_batch_size,
                token_budget=self.microbatch_tokens,
            ):
                width = max(len(b[0]) for b in group)
                ids = self.engine.tensor(
                    [
                        b + [self.engine.tokenizer.pad_token_id] * (width - len(b))
                        for b, _, _ in group
                    ]
                )
                mask = self.engine.tensor(
                    [[1] * (len(prefix) + len(b)) + [0] * (width - len(b)) for b, _, _ in group]
                )
                output = self.engine.model(
                    ids,
                    past_key_values=self.native.fork_cache(cache, len(group)),
                    attention_mask=mask,
                    use_cache=True,
                )
                for row, (_, start, value) in enumerate(group):
                    logp = (
                        output.logits[row, start - 1 : start + len(value) - 1]
                        .float()
                        .log_softmax(-1)
                    )
                    score = logp.gather(1, self.engine.tensor(value)[:, None]).sum()
                    scores.append(float(score.cpu()))
                del output, ids, mask
        return scores

    def predict(self, case):
        predictions = []
        for question in case.questions:
            context, schema, _ = rlcd_inputs(case.state, question)
            scores = self.branch_scores(context, schema)
            if len(scores) != len(question.options):
                raise ValueError("RLCD returned incorrect candidate count")
            probabilities = self.torch.tensor(scores, dtype=self.torch.float64).softmax(-1).tolist()
            raw = dict(zip([o.id for o in question.options], probabilities, strict=True))
            check_probabilities(raw, [o.id for o in question.options])
            predictions.append(
                Prediction(case_id=case.case_id, question_id=question.id, probabilities=raw)
            )
        return predictions
