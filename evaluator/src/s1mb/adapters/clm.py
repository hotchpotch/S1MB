"""CLM's native typed renderer and projection heads with local Qwen embeddings."""

import importlib
from typing import Any

from .base import decode_answers
from .firelex_jeff import decision_row
from .upstream import UpstreamAdapter, candidate_batches, checkpoint_path

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def last_token_embeddings(hidden, attention_mask):
    """Select the final non-padding token; support either padding direction."""
    torch = importlib.import_module("torch")
    positions = torch.arange(attention_mask.shape[1], device=hidden.device)
    last = positions.masked_fill(~attention_mask.bool(), -1).amax(dim=1)
    if (last < 0).any():
        raise ValueError("CLM cannot embed an empty token sequence")
    pooled = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
    return torch.nn.functional.normalize(pooled, dim=-1)


class CLMAdapter(UpstreamAdapter):
    """Use full inputs, fixed native pooling and FP32 contrastive projection heads."""

    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768, case_batch_size=1):
        if case_batch_size != 1:
            raise ValueError("CLM currently requires case_batch_size=1")
        self.setup("clm", model, revision, source, device)
        if not 0 < context_limit <= 32768:
            raise ValueError("CLM context_limit must be between 1 and 32768")
        self.context_limit = context_limit
        transformers = importlib.import_module("transformers")
        self.schema = importlib.import_module("clm.schema")
        heads = importlib.import_module("clm.heads")
        base_path, base_revision = checkpoint_path(BASE_MODEL, BASE_REVISION)
        self.tokenizer: Any = transformers.AutoTokenizer.from_pretrained(base_path)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.engine = (
            transformers.AutoModel.from_pretrained(
                base_path, dtype=self.torch.bfloat16, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        self.attention_model = self.engine
        self.head = heads.HeadPair("clm", str(self.path / "CLM_v0.1-8B.pt"), device).ensure()
        # Include both heads in parameter metadata and normal module cleanup.
        self.engine.add_module("clm_state_head", self.head.state_head)
        self.engine.add_module("clm_action_head", self.head.action_head)
        self.reset_action_cache()
        self.settings = {
            "base_model": BASE_MODEL,
            "base_revision": base_revision,
            "dtype": "bfloat16",
            "projection_dtype": "float32",
            "attention_implementation": "sdpa",
            "max_length": context_limit,
            "input_length_policy": "reject-overflow",
            "pooling": "last-nonpadding-token-l2-normalized-float32",
            "renderer": "native-clm-structured-anonymous-choice-numeric-score-v2",
            "probability_rule": "native-scaled-cosine-softmax",
            "logit_scale": self.head.scale,
            "case_batch_size": self.case_batch_size,
            "microbatch_tokens": 4096,
            "embedding_batch_size": 1,
            "embedding_cache": "within-call-deduplication-only",
            "action_projection_cache": "native-vector-arena-exact-text-and-head-namespace",
            "action_cache_budget_bytes": 64 << 20,
            "action_cache_reserved_bytes": self.action_cache.reserved_bytes,
            "state_cache": False,
        }

    def embed(self, texts):
        """Return normalized embeddings without truncation or cross-call caching."""
        unique = list(dict.fromkeys(texts))
        encoded = self.tokenizer(unique, truncation=False)["input_ids"]
        lengths = [len(tokens) for tokens in encoded]
        if any(length == 0 or length > self.context_limit for length in lengths):
            raise ValueError(f"CLM input token length outside [1, {self.context_limit}]")
        vectors = []
        with self.torch.inference_mode():
            for indices in candidate_batches(range(len(unique)), lengths, max_batch=1):
                batch = self.tokenizer.pad(
                    {"input_ids": [encoded[i] for i in indices]}, return_tensors="pt"
                ).to(self.device)
                hidden = self.engine(**batch, use_cache=False).last_hidden_state
                vectors.extend(last_token_embeddings(hidden, batch["attention_mask"]).unbind())
        lookup = dict(zip(unique, vectors, strict=True))
        return self.torch.stack([lookup[text] for text in texts])

    def reset_action_cache(self):
        """Start a bounded, empty action cache independently of model warmup."""
        if hasattr(self, "action_cache"):
            del self.action_cache
        cache = importlib.import_module("clm.cache")
        self.action_cache = cache.VectorArena(device=self.device, budget="64MiB")
        self.action_cache.reserve(self.head.proj_dim, 1.0)

    def action_vectors(self, texts):
        """Cache only GPU action projections; state inference remains request-local."""

        def compute(missing):
            return self.torch.nn.functional.normalize(
                self.head.action_head(self.embed(missing)), dim=-1
            )

        return self.action_cache.get(
            f"{self.head.namespace}/action", self.head.proj_dim, texts, compute
        )

    def predict(self, case):
        questions = {q.id: decision_row(case.state, q)["question"] for q in case.questions}
        pairs = self.schema.build_pairs(case.state, questions)
        states = [pair[0] for pair in pairs.values()]
        actions = [text for pair in pairs.values() for text in pair[2]]
        with self.torch.inference_mode():
            state_embeddings = self.embed(states)
            normalize = self.torch.nn.functional.normalize
            state_vectors = normalize(self.head.state_head(state_embeddings), dim=-1)
            action_vectors = self.action_vectors(actions)
            answers, offset = {}, 0
            for i, (qid, (_, keys, texts)) in enumerate(pairs.items()):
                logits = self.head.scale * (
                    action_vectors[offset : offset + len(texts)] @ state_vectors[i]
                )
                offset += len(texts)
                answers[qid] = self.schema.answer_from_logits(questions[qid], keys, logits.tolist())
        return decode_answers(case, answers, anonymous_choice=True)

    def predict_batch(self, cases):
        return [self.predict(case) for case in cases]

    def close(self):
        if hasattr(self, "action_cache"):
            del self.action_cache
        if hasattr(self, "head"):
            del self.head
        super().close()
