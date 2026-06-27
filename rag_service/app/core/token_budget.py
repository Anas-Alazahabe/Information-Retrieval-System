"""Character limits for RAG context assembly."""

from typing import Dict, List, Tuple

DEFAULT_MAX_CHARS_PER_DOC = 2000


def apply_char_budget(
    ranked_docs: List[Tuple[str, float, str]],
    *,
    max_context_chars: int,
    max_chars_per_doc: int = DEFAULT_MAX_CHARS_PER_DOC,
) -> Tuple[List[Tuple[str, float, str]], int]:
    """Trim passages to fit global and per-document char budgets.

    Drops lowest-scored documents first when over budget.
    Returns (kept_docs, total_chars).
    """
    if not ranked_docs or max_context_chars <= 0:
        return [], 0

    trimmed: List[Tuple[str, float, str]] = []
    for doc_id, score, text in ranked_docs:
        snippet = text[:max_chars_per_doc] if len(text) > max_chars_per_doc else text
        trimmed.append((doc_id, score, snippet))

    while trimmed:
        total = sum(len(text) for _, _, text in trimmed)
        if total <= max_context_chars:
            return trimmed, total
        trimmed.pop()

    return [], 0
