import time
from typing import Optional

from shared.ir_config import EMBEDDING_SEARCH_K

from retrieval_service.app.core.search_engine import EmbeddingSearchEngine

from .base import MatchParams, MatchResult, QueryRepresentation


def _embedding_search_k(params: MatchParams, for_rrf: bool = False) -> int:
    """Candidate pool size for ANN/full embedding search before final top_k truncation."""
    if for_rrf:
        return EMBEDDING_SEARCH_K
    return params.top_k if params.top_k is not None else EMBEDDING_SEARCH_K


class EmbeddingMatcher:
    """Dense embedding matching via cosine similarity."""

    mode = "embedding"
    matching_method = "cosine_similarity"

    def __init__(self, engine: EmbeddingSearchEngine):
        self.engine = engine

    def match(self, query: QueryRepresentation, params: MatchParams) -> MatchResult:
        t0 = time.perf_counter()
        if not query.raw_text.strip():
            return MatchResult(scores={}, match_ms=0.0, empty_reason="empty_query")

        if not self.engine.doc_embeddings:
            return MatchResult(
                scores={},
                match_ms=round((time.perf_counter() - t0) * 1000, 2),
                empty_reason="no_embeddings_indexed",
            )

        encode_ms = 0.0
        self.engine._lazy_load_model()
        if not self.engine.model:
            return MatchResult(
                scores={},
                match_ms=round((time.perf_counter() - t0) * 1000, 2),
                empty_reason="embedding_model_unavailable",
            )

        t_encode = time.perf_counter()
        encoded = self.engine.model.encode(query.raw_text, normalize_embeddings=True)
        query_vector = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        encode_ms = round((time.perf_counter() - t_encode) * 1000, 2)

        t_match = time.perf_counter()
        scores = self.engine._score_vectors(
            query_vector, doc_ids=None, top_k=_embedding_search_k(params)
        )
        match_ms = round((time.perf_counter() - t_match) * 1000, 2)

        return MatchResult(
            scores=scores,
            match_ms=match_ms,
            encode_ms=encode_ms,
        )

    def match_candidates(
        self,
        query: QueryRepresentation,
        doc_ids: list,
        params: Optional[MatchParams] = None,
    ) -> MatchResult:
        """Score a fixed candidate set (used by serial hybrid)."""
        t0 = time.perf_counter()
        if not doc_ids:
            return MatchResult(scores={}, match_ms=0.0, empty_reason="no_candidates")

        self.engine._lazy_load_model()
        if not self.engine.model:
            return MatchResult(
                scores={},
                match_ms=round((time.perf_counter() - t0) * 1000, 2),
                empty_reason="embedding_model_unavailable",
            )

        t_encode = time.perf_counter()
        encoded = self.engine.model.encode(query.raw_text, normalize_embeddings=True)
        query_vector = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        encode_ms = round((time.perf_counter() - t_encode) * 1000, 2)

        t_match = time.perf_counter()
        scores = self.engine._score_vectors(query_vector, doc_ids=doc_ids)
        match_ms = round((time.perf_counter() - t_match) * 1000, 2)

        return MatchResult(
            scores=scores,
            match_ms=match_ms,
            encode_ms=encode_ms,
        )
