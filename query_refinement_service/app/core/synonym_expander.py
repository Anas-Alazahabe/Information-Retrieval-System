"""WordNet-based query synonym expansion."""

import logging
from typing import FrozenSet, List, Optional, Set

import nltk
from nltk.corpus import stopwords, wordnet

from shared.ir_config import SYNONYM_MAX_PER_TERM, SYNONYM_MAX_TOTAL, WH_WORDS

logger = logging.getLogger(__name__)

_wordnet_ready = False
_ENGLISH_STOPWORDS = frozenset(stopwords.words("english"))


def _ensure_wordnet() -> None:
    """Download WordNet corpora on first use."""
    global _wordnet_ready
    if _wordnet_ready:
        return
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)
    _wordnet_ready = True


def _normalize_lemma(lemma: str) -> Optional[str]:
    """Normalize a WordNet lemma to a single lowercase token."""
    token = lemma.lower().replace("_", " ").strip()
    if " " in token:
        return None
    if len(token) < 2 or not token.isalpha():
        return None
    return token


def _synonyms_for_token(token: str) -> List[str]:
    """Collect normalized synonym lemmas for one query token."""
    _ensure_wordnet()
    candidates: Set[str] = set()
    for syn in wordnet.synsets(token):
        for lemma in syn.lemmas():
            normalized = _normalize_lemma(lemma.name())
            if normalized and normalized != token:
                candidates.add(normalized)
    return sorted(candidates)


def expand_synonyms(
    tokens: List[str],
    max_synonyms_per_term: int = SYNONYM_MAX_PER_TERM,
    max_total: int = SYNONYM_MAX_TOTAL,
    skip_terms: Optional[FrozenSet[str]] = None,
) -> List[str]:
    """Expand query tokens with English WordNet synonyms (bounded, deterministic)."""
    if not tokens or max_total <= 0:
        return []

    skip = skip_terms or WH_WORDS
    skip = frozenset(skip) | _ENGLISH_STOPWORDS
    existing = {t.lower() for t in tokens}
    expanded: List[str] = []
    seen: Set[str] = set(existing)

    for token in tokens:
        if len(expanded) >= max_total:
            break
        lower = token.lower()
        if lower in skip:
            continue

        per_term_added = 0
        for synonym in _synonyms_for_token(lower):
            if len(expanded) >= max_total or per_term_added >= max_synonyms_per_term:
                break
            if synonym in seen:
                continue
            expanded.append(synonym)
            seen.add(synonym)
            per_term_added += 1

    return expanded
