"""Query matching and ranking contracts (Task 6).

Query and document representations must stay consistent:
- VSM/BM25: same preprocessed tokens and index-time weighting (log-TF-IDF / BM25).
- Embedding: same ``EMBEDDING_MODEL`` as used at index time.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class QueryRepresentation:
    """Unified query payload passed from preprocessing to matchers."""

    raw_text: str
    tokens: List[str]
    mode: str


@dataclass
class MatchParams:
    """Per-request matching parameters."""

    k1: float = 1.5
    b: float = 0.75
    top_n_filter: int = 100
    k_rrf: int = 60
    top_k: Optional[int] = None


@dataclass
class MatchResult:
    """Raw scores from a matcher before final ranking truncation."""

    scores: Dict[str, float] = field(default_factory=dict)
    match_ms: float = 0.0
    encode_ms: float = 0.0
    empty_reason: Optional[str] = None


@runtime_checkable
class BaseMatcher(Protocol):
    """Protocol for representation-specific query–document matching."""

    mode: str
    matching_method: str

    def match(self, query: QueryRepresentation, params: MatchParams) -> MatchResult: ...


class Ranker:
    """Sort scores descending with deterministic tie-breaking."""

    @staticmethod
    def rank(scores: Dict[str, float], top_k: Optional[int]) -> Dict[str, float]:
        if not scores:
            return {}
        sorted_items = sorted(
            scores.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
        if top_k is not None and top_k > 0:
            sorted_items = sorted_items[:top_k]
        return dict(sorted_items)
