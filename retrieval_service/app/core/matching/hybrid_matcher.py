import time
from dataclasses import replace

from shared.ir_config import EMBEDDING_SEARCH_K

from retrieval_service.app.core.search_engine import BM25SearchEngine, EmbeddingSearchEngine
from .base import MatchParams, MatchResult, QueryRepresentation
from .embedding_matcher import EmbeddingMatcher


class HybridParallelMatcher:
    """Parallel hybrid: RRF fusion of BM25 and embedding rank lists."""

    mode = "hybrid_parallel"
    matching_method = "rrf"

    def __init__(
        self,
        bm25_engine: BM25SearchEngine,
        embedding_engine: EmbeddingSearchEngine,
    ):
        self.bm25_engine = bm25_engine
        self.embedding_matcher = EmbeddingMatcher(embedding_engine)

    def match(self, query: QueryRepresentation, params: MatchParams) -> MatchResult:
        t0 = time.perf_counter()
        bm25_results = self.bm25_engine.search(query.tokens, k1=params.k1, b=params.b)
        emb_result = self.embedding_matcher.match(
            query, replace(params, top_k=EMBEDDING_SEARCH_K)
        )
        embedding_results = emb_result.scores

        bm25_rank_list = list(bm25_results.keys())
        emb_rank_list = list(embedding_results.keys())
        bm25_ranks = {doc_id: i + 1 for i, doc_id in enumerate(bm25_rank_list)}
        emb_ranks = {doc_id: i + 1 for i, doc_id in enumerate(emb_rank_list)}

        all_docs = set(bm25_rank_list).union(set(emb_rank_list))
        rrf_scores = {}
        k_rrf = params.k_rrf

        for doc_id in all_docs:
            rank_bm25 = bm25_ranks.get(doc_id)
            rank_emb = emb_ranks.get(doc_id)
            score_bm25 = (
                params.bm25_rrf_weight / (k_rrf + rank_bm25) if rank_bm25 else 0.0
            )
            score_emb = (
                params.embedding_rrf_weight / (k_rrf + rank_emb) if rank_emb else 0.0
            )
            rrf_scores[doc_id] = round(score_bm25 + score_emb, 6)

        sorted_scores = dict(
            sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        )
        empty_reason = None
        if not sorted_scores:
            empty_reason = emb_result.empty_reason or "no_lexical_overlap"

        return MatchResult(
            scores=sorted_scores,
            match_ms=round((time.perf_counter() - t0) * 1000, 2),
            encode_ms=emb_result.encode_ms,
            empty_reason=empty_reason,
        )


class HybridSerialMatcher:
    """Serial hybrid: BM25 candidate filter then embedding cosine rerank."""

    mode = "hybrid_serial"
    matching_method = "bm25_filter_cosine_rerank"

    def __init__(
        self,
        bm25_engine: BM25SearchEngine,
        embedding_engine: EmbeddingSearchEngine,
    ):
        self.bm25_engine = bm25_engine
        self.embedding_matcher = EmbeddingMatcher(embedding_engine)

    def match(self, query: QueryRepresentation, params: MatchParams) -> MatchResult:
        t0 = time.perf_counter()
        bm25_results = self.bm25_engine.search(query.tokens, k1=params.k1, b=params.b)
        top_docs = list(bm25_results.keys())[: params.top_n_filter]

        if not top_docs:
            return MatchResult(
                scores={},
                match_ms=round((time.perf_counter() - t0) * 1000, 2),
                empty_reason="no_lexical_overlap",
            )

        emb_result = self.embedding_matcher.match_candidates(query, top_docs, params)
        if emb_result.scores:
            return MatchResult(
                scores=emb_result.scores,
                match_ms=round((time.perf_counter() - t0) * 1000, 2),
                encode_ms=emb_result.encode_ms,
            )

        return MatchResult(
            scores=bm25_results,
            match_ms=round((time.perf_counter() - t0) * 1000, 2),
            encode_ms=emb_result.encode_ms,
            empty_reason=emb_result.empty_reason,
        )
