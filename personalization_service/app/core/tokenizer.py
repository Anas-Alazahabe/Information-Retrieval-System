"""Lightweight tokenization for personalization profiles."""

import re
from typing import List

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

PROFILE_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "is", "it", "for", "on", "at",
    "as", "by", "be", "has", "was", "are", "with", "that", "this", "from", "you",
    "your", "can", "will", "not", "but", "they", "their", "we", "our", "have", "had",
    "been", "were", "which", "what", "when", "where", "how", "who", "also", "than",
    "into", "about", "over", "after", "before", "between", "through", "during",
    "do", "does", "did", "if", "so", "no", "yes", "up", "out", "all", "any",
})


def lightweight_tokenize(text: str, *, drop_stopwords: bool = True) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.strip()).lower()
    if not normalized:
        return []
    tokens = _TOKEN_PATTERN.findall(normalized)
    if drop_stopwords:
        tokens = [
            t
            for t in tokens
            if t not in PROFILE_STOPWORDS and len(t) > 1 and not t.isdigit()
        ]
    return tokens
