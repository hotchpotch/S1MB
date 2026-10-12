"""Explicit bounded state shortening that preserves structured decision fields."""

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PROTECTED_FIELDS = {
    "question",
    "questions",
    "instruction",
    "instructions",
    "criteria",
    "candidates",
    "options",
    "id",
    "identifier",
    "title",
    "name",
    "authors",
}


@dataclass
class BudgetedState:
    state: Any
    request_before: int
    request_after: int
    state_before: int
    state_after: int


def budget_state(
    state,
    tokenizer,
    render: Callable[[Any], str],
    measure: Callable[[Any], int],
    limit: int,
    max_reduction: float,
    min_retained_tokens: int = 1,
) -> BudgetedState:
    """Shorten long body strings, preserving JSON structure and head/tail context.

    The fraction bounds removed tokens in the rendered state, not the model's
    full context capacity. Instructions, candidates and question fields remain
    intact. Inputs beyond this explicit allowance are rejected.
    """
    if not math.isfinite(max_reduction) or not 0 < max_reduction <= 1:
        raise ValueError("State token reduction must be within (0, 1]")

    if type(min_retained_tokens) is not int or min_retained_tokens < 1:
        raise ValueError("Minimum retained text must be a positive token count")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    original_tokens = len(encode(render(state)))
    before = measure(state)
    if before <= limit:
        return BudgetedState(state, before, before, original_tokens, original_tokens)
    allowance = math.floor(original_tokens * max_reduction)
    if before - limit > allowance:
        raise ValueError("Input exceeds the explicitly allowed state token reduction")
    working = copy.deepcopy(state)
    leaves = []

    def collect(value, path=()):
        if isinstance(value, str):
            ids = encode(value)
            if len(ids) >= 32:
                leaves.append((path, ids))
        elif isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() not in PROTECTED_FIELDS:
                    collect(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, (*path, index))

    collect(working)
    leaves.sort(key=lambda leaf: len(leaf[1]), reverse=True)
    if not leaves:
        raise ValueError("No sufficiently long unprotected body text can be shortened")

    def assign(path, value):
        nonlocal working
        if not path:
            working = value
        else:
            parent = working
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = value

    # Bounded search verifies the complete native rendering on every attempt.
    for path, ids in leaves:
        baseline_tokens = len(encode(render(working)))
        low, high = 0, len(ids) - min_retained_tokens
        accepted = None
        best_trial = None
        while low <= high:
            removed = (low + high) // 2
            retained = len(ids) - removed
            head = (retained + 1) // 2
            tail = retained // 2
            text = tokenizer.decode(ids[:head], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            text += "\n[…]\n" + tokenizer.decode(
                ids[-tail:] if tail else [], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            assign(path, text)
            state_after = len(encode(render(working)))
            removed_total = original_tokens - state_after
            if removed_total > allowance:
                high = removed - 1
                continue
            after = measure(working)
            if best_trial is None or state_after < best_trial[2]:
                best_trial = (copy.deepcopy(working), after, state_after)
            if after <= limit:
                accepted = (copy.deepcopy(working), after, state_after)
                high = removed - 1
            else:
                low = removed + 1
        if accepted is not None:
            content, after, state_after = accepted
            return BudgetedState(content, before, after, original_tokens, state_after)
        if best_trial is not None and best_trial[2] < baseline_tokens:
            working = best_trial[0]
        else:
            original = state
            for key in path:
                original = original[key]
            assign(path, original)
    raise ValueError("Input cannot fit without exceeding the allowed state token reduction")
