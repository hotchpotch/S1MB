"""Dedicated relevance checkpoints mapped to explicit candidate distributions."""

import importlib
import json

from s1mb.data import Prediction, check_probabilities

from .base import questions_for_api
from .upstream import UpstreamAdapter, candidate_batches, state_text


def reranker_inputs(state, question):
    item = questions_for_api([question], structured=True, anonymous_choice=True)[question.id]
    instruction = state_text(item["instructions"])
    documents = []
    for index, option in enumerate(question.options):
        label = (
            option.id
            if question.task == "noul"
            else str(option.value)
            if question.task == "score"
            else f"option_{index}"
        )
        documents.append(
            f"{label}: {option.description_json if option.description_json is not None else option.description}"
        )
    query = (
        f"Situation:\n{state_text(state)}\n\nQuestion:\n{instruction}\n\nAnswer rubric:\n"
        + "\n".join(documents)
    )
    return instruction, query, documents


class RerankerAdapter(UpstreamAdapter):
    case_batch_size = 1
    candidate_batch_size = 8
    microbatch_tokens = 8192

    def __init__(self, model, revision, source, device, context_limit=None):
        self.setup("reranker", model, revision, source, device)
        transformers = importlib.import_module("transformers")
        config = transformers.AutoConfig.from_pretrained(str(self.path), trust_remote_code=True)
        self.causal = any("CausalLM" in name for name in config.architectures)
        if self.causal:
            # BF16 candidate logits changed materially with padded batching on
            # Qwen3. Preserve single-pair inference for decoder rerankers.
            self.candidate_batch_size = 1
        cls = (
            transformers.AutoModelForCausalLM
            if self.causal
            else transformers.AutoModelForSequenceClassification
        )
        self.engine = (
            cls.from_pretrained(
                str(self.path),
                trust_remote_code=True,
                dtype=self.torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(self.path), padding_side="left" if self.causal else "right"
        )
        if self.tokenizer is None:
            raise ValueError("Checkpoint did not provide a tokenizer")
        self.context_limit = context_limit or min(
            getattr(config, "max_position_embeddings", 8192), 8192
        )
        requested_limit = self.context_limit
        # XLM-R position IDs start after its padding index. Its learned table
        # cannot be extended by merely requesting a larger runtime context.
        if config.model_type == "xlm-roberta":
            self.context_limit = min(
                self.context_limit, config.max_position_embeddings - config.pad_token_id - 1
            )
        logit_config = self.path / "1_LogitScore/config.json"
        self.logit_score = json.loads(logit_config.read_text()) if logit_config.exists() else None
        if self.causal and not self.logit_score:
            if config.model_type != "qwen3":
                raise ValueError("Only the published Qwen3 causal reranker format is supported")
            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.prefix = self.tokenizer.encode(prefix, add_special_tokens=False)
            self.suffix = self.tokenizer.encode(suffix, add_special_tokens=False)
            self.no_id = self.tokenizer.convert_tokens_to_ids("no")
            self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.settings = {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_input_tokens": self.context_limit,
            "requested_context_limit": requested_limit,
            "input_length_policy": "reject-overflow",
            "checkpoint_position_limit": getattr(config, "max_position_embeddings", None),
            "case_batch_size": 1,
            "candidate_batch_size": self.candidate_batch_size,
            "microbatch_tokens": self.microbatch_tokens,
            "probability_origin": "softmax-of-native-relevance-scores",
            "native_relevance": "native-token-logit"
            if self.logit_score
            else "yes-probability"
            if self.causal
            else "classification-logit",
            "logit_score": self.logit_score,
            "temperature": 1.0,
            "renderer": "full-rubric-anonymous-choice-numeric-score-v1",
        }

    def predict(self, case):
        predictions = []
        for question in case.questions:
            instruction, query, documents = reranker_inputs(case.state, question)
            scores = []
            encoded = []
            for document in documents:
                if self.causal:
                    if self.logit_score:
                        ids = self.tokenizer.apply_chat_template(
                            [
                                {"role": "query", "content": query},
                                {"role": "document", "content": document},
                            ],
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=False,
                        )
                    else:
                        text = (
                            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
                        )
                        ids = (
                            self.prefix
                            + self.tokenizer.encode(text, add_special_tokens=False)
                            + self.suffix
                        )
                    row = {"input_ids": ids, "attention_mask": [1] * len(ids)}
                else:
                    row = self.tokenizer(query, document, truncation=False)
                if len(row["input_ids"]) > self.context_limit:
                    raise ValueError("Reranker input exceeds context limit; refusing truncation")
                encoded.append(row)
            for batch in candidate_batches(
                encoded,
                [len(row["input_ids"]) for row in encoded],
                self.candidate_batch_size,
                self.microbatch_tokens,
            ):
                inputs = self.tokenizer.pad(batch, padding=True, return_tensors="pt").to(
                    self.device
                )
                with self.torch.inference_mode():
                    kwargs = {"logits_to_keep": 1} if self.causal else {}
                    logits = self.engine(**inputs, **kwargs).logits.float()
                    if self.logit_score:
                        value = logits[:, -1, self.logit_score["true_token_id"]]
                        false_id = self.logit_score.get("false_token_id")
                        if false_id is not None:
                            value = value - logits[:, -1, false_id]
                        scores.append(value)
                    elif self.causal:
                        scores.append(logits[:, -1, [self.no_id, self.yes_id]].softmax(-1)[:, 1])
                    else:
                        if logits.numel() != len(batch):
                            raise ValueError("Reranker must emit one relevance logit per pair")
                        scores.append(logits.reshape(-1))
            values = self.torch.cat(scores).softmax(-1).cpu().tolist()
            probabilities = dict(zip([o.id for o in question.options], values, strict=True))
            check_probabilities(probabilities, [o.id for o in question.options])
            predictions.append(
                Prediction(
                    case_id=case.case_id, question_id=question.id, probabilities=probabilities
                )
            )
        return predictions
