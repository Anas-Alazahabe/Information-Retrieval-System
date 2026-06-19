import math
import time
from collections import Counter
from typing import Any, Dict, MutableMapping

from shared.index_store import IndexStore

from .base import MatchParams, MatchResult, QueryRepresentation


class VSMMatcher:
    """VSM TF-IDF matching via cosine similarity."""

    mode = "vsm"
    matching_method = "cosine_similarity"

    def __init__(self, store: IndexStore, vsm_cache: MutableMapping[str, Any]):
        self.store = store
        self._vsm_cache = vsm_cache

    def _load_vsm_data(self):
        if "vsm_index" not in self._vsm_cache:
            self._vsm_cache["vsm_index"] = self.store.load_vsm()
            self._vsm_cache["metadata"] = self.store.load_metadata()
        return self._vsm_cache["vsm_index"], self._vsm_cache["metadata"]

    def _doc_norms(self, vsm_index: Dict, metadata: Dict) -> Dict[str, float]:
        doc_norms = metadata.get("doc_norms", {})
        if doc_norms:
            return doc_norms
        doc_weight_squares: Dict[str, float] = {}
        for postings in vsm_index.values():
            for doc_id, weight in postings.items():
                doc_weight_squares[doc_id] = doc_weight_squares.get(doc_id, 0.0) + weight ** 2
        return {
            doc_id: math.sqrt(square_sum)
            for doc_id, square_sum in doc_weight_squares.items()
            if square_sum > 0
        }

    def match(self, query: QueryRepresentation, params: MatchParams) -> MatchResult:
        t0 = time.perf_counter()
        vsm_index, metadata = self._load_vsm_data()
        idf_weights = metadata.get("idf_weights", {})
        doc_norms = self._doc_norms(vsm_index, metadata)

        query_counts = Counter(query.tokens)
        query_vector: Dict[str, float] = {}
        query_norm = 0.0

        for term, count in query_counts.items():
            if term in idf_weights:
                idf = idf_weights[term]
                log_tf = 1 + math.log10(count) if count > 0 else 0
                weight = log_tf * idf
                query_vector[term] = weight
                query_norm += weight ** 2

        query_norm = math.sqrt(query_norm)
        if query_norm == 0.0:
            return MatchResult(
                scores={},
                match_ms=round((time.perf_counter() - t0) * 1000, 2),
                empty_reason="no_lexical_overlap",
            )

        doc_scores: Dict[str, float] = {}
        for term, q_weight in query_vector.items():
            if term in vsm_index:
                for doc_id, doc_weight in vsm_index[term].items():
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (q_weight * doc_weight)

        final_scores: Dict[str, float] = {}
        for doc_id, score in doc_scores.items():
            doc_norm = doc_norms.get(doc_id, 0.0)
            if doc_norm <= 0.0:
                continue
            final_scores[doc_id] = round(score / (query_norm * doc_norm), 4)

        sorted_scores = dict(
            sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        )
        empty_reason = "no_lexical_overlap" if not sorted_scores else None
        return MatchResult(
            scores=sorted_scores,
            match_ms=round((time.perf_counter() - t0) * 1000, 2),
            empty_reason=empty_reason,
        )
