from typing import Any, Dict, MutableMapping, Optional

from shared.index_store import IndexStore
from shared.ir_config import MATCHER_METADATA, RRF_K, SERIAL_HYBRID_TOP_N

from retrieval_service.app.core.matching.base import BaseMatcher, MatchParams
from retrieval_service.app.core.matching.bm25_matcher import BM25Matcher
from retrieval_service.app.core.matching.embedding_matcher import EmbeddingMatcher
from retrieval_service.app.core.matching.hybrid_matcher import HybridParallelMatcher, HybridSerialMatcher
from retrieval_service.app.core.matching.vsm_matcher import VSMMatcher
from retrieval_service.app.core.search_engine import BM25SearchEngine, EmbeddingSearchEngine


class MatcherRegistry:
    """Builds and caches matcher instances bound to shared engines."""

    def __init__(
        self,
        store: IndexStore,
        bm25_engine: BM25SearchEngine,
        embedding_engine: EmbeddingSearchEngine,
        vsm_cache: MutableMapping[str, Any],
    ):
        self.store = store
        self.bm25_engine = bm25_engine
        self.embedding_engine = embedding_engine
        self.vsm_cache = vsm_cache
        self._matchers: Dict[str, BaseMatcher] = {}

    def _build(self, mode: str) -> BaseMatcher:
        if mode == "vsm":
            return VSMMatcher(self.store, self.vsm_cache)
        if mode == "bm25":
            return BM25Matcher(self.bm25_engine)
        if mode == "embedding":
            return EmbeddingMatcher(self.embedding_engine)
        if mode == "hybrid_parallel":
            return HybridParallelMatcher(self.bm25_engine, self.embedding_engine)
        if mode == "hybrid_serial":
            return HybridSerialMatcher(self.bm25_engine, self.embedding_engine)
        raise ValueError(f"Unknown matcher mode: {mode}")

    def get(self, mode: str) -> BaseMatcher:
        if mode not in self._matchers:
            self._matchers[mode] = self._build(mode)
        return self._matchers[mode]

    def clear_vsm_cache(self):
        self.vsm_cache.clear()


def default_match_params(
    k1: float = 1.5,
    b: float = 0.75,
    top_n_filter: Optional[int] = None,
    k_rrf: Optional[int] = None,
    top_k: Optional[int] = None,
) -> MatchParams:
    return MatchParams(
        k1=k1,
        b=b,
        top_n_filter=top_n_filter if top_n_filter is not None else SERIAL_HYBRID_TOP_N,
        k_rrf=k_rrf if k_rrf is not None else RRF_K,
        top_k=top_k,
    )


def list_matchers() -> list:
    """Return metadata for all supported matchers (GET /matchers)."""
    return [
        {"mode": mode, **meta}
        for mode, meta in MATCHER_METADATA.items()
    ]
