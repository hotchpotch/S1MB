"""Reject silent truncation when evaluating with explicit branch length limits."""


def check_balanced_input_lengths_batched(tokenizer, parts, query_length):
    """Validate every field without truncation, tokenizing repeated text only once."""
    fields = [
        [
            "Instruction: ",
            "State: ",
            "\n",
            f"{part.system}\n\n" if part.system.strip() else "",
            part.instruction,
            part.context,
        ]
        for part in parts
    ]
    unique = list(dict.fromkeys(text for group in fields for text in group))
    if not unique:
        return
    encoded = tokenizer(unique, add_special_tokens=False, truncation=False)["input_ids"]
    lengths = dict(zip(unique, map(len, encoded), strict=True))
    for group in fields:
        required = 2 + sum(lengths[text] for text in group)
        if required > query_length:
            raise ValueError(f"Balanced query requires {required} tokens; refusing truncation")


def check_input_lengths(tokenizer, queries, documents, query_length, document_length):
    for name, texts, limit, special in (
        ("query", queries, query_length, 2),
        ("candidate", documents, document_length, 1),
    ):
        unique = list(dict.fromkeys(texts))
        if not unique:
            continue
        ids = tokenizer(unique, add_special_tokens=False, truncation=False)["input_ids"]
        maximum = max(len(tokens) + special for tokens in ids)
        if maximum > limit:
            raise ValueError(
                f"{name} requires {maximum} tokens, exceeding {limit}; refusing truncation"
            )
