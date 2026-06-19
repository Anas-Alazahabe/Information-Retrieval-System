"""Session history enrichment for query refinement."""

import re
from typing import Dict, List, Tuple

from shared.ir_config import HISTORY_MAX_QUERIES, HISTORY_MAX_TERMS

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _lightweight_tokenize(text: str) -> List[str]:
    normalized = _normalize_query(text).lower()
    return _TOKEN_PATTERN.findall(normalized)


def merge_history_terms(
    current_query: str,
    current_tokens: List[str],
    previous_queries: List[str],
    *,
    max_terms: int = HISTORY_MAX_TERMS,
    max_queries: int = HISTORY_MAX_QUERIES,
) -> Tuple[List[str], str]:
    """Merge salient terms from prior session queries into the current query."""
    if not previous_queries:
        return [], "No session history provided."

    current_lower = {token.lower() for token in current_tokens}
    current_lower.update(_lightweight_tokenize(current_query))

    normalized_current = _normalize_query(current_query).lower()
    prior_queries: List[str] = []
    for query in reversed(previous_queries):
        normalized = _normalize_query(query)
        if not normalized:
            continue
        if normalized.lower() == normalized_current:
            continue
        if normalized in prior_queries:
            continue
        prior_queries.append(normalized)
        if len(prior_queries) >= max_queries:
            break

    if not prior_queries:
        return [], "No usable session history queries."

    term_scores: Dict[str, float] = {}
    term_sources: Dict[str, str] = {}

    for index, query in enumerate(prior_queries):
        weight = 1.0 / (2 ** index)
        for token in _lightweight_tokenize(query):
            if token in current_lower:
                continue
            term_scores[token] = term_scores.get(token, 0.0) + weight
            if token not in term_sources:
                term_sources[token] = query

    if not term_scores:
        return [], "Session history had no new terms to add."

    ranked_terms = sorted(term_scores.items(), key=lambda item: (-item[1], item[0]))
    selected = [token for token, _ in ranked_terms[:max_terms]]

    source_snippets = []
    seen_sources = set()
    for token in selected:
        source = term_sources[token]
        if source not in seen_sources:
            source_snippets.append(f"'{source}'")
            seen_sources.add(source)

    explanation = (
        f"History added: {', '.join(selected)} "
        f"(from {', '.join(source_snippets)})."
    )
    return selected, explanation
