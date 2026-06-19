import time

from retrieval_service.app.core.search_engine import BM25SearchEngine

from .base import MatchParams, MatchResult, QueryRepresentation


class BM25Matcher:
    """BM25 lexical matching."""

    mode = "bm25"
    matching_method = "bm25"

    def __init__(self, engine: BM25SearchEngine):
        self.engine = engine

    def match(self, query: QueryRepresentation, params: MatchParams) -> MatchResult:
        t0 = time.perf_counter()
        if not query.tokens:
            return MatchResult(
                scores={},
                match_ms=round((time.perf_counter() - t0) * 1000, 2),
                empty_reason="no_lexical_overlap",
            )

        scores = self.engine.search(query.tokens, k1=params.k1, b=params.b)
        empty_reason = "no_lexical_overlap" if not scores else None
        return MatchResult(
            scores=scores,
            match_ms=round((time.perf_counter() - t0) * 1000, 2),
            empty_reason=empty_reason,
        )
