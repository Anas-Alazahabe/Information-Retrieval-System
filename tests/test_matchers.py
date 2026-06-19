import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "matcher_index"
sys.path.insert(0, str(ROOT))

from retrieval_service.app.core.matching.base import MatchParams, QueryRepresentation, Ranker
from retrieval_service.app.core.matching.bm25_matcher import BM25Matcher
from retrieval_service.app.core.matching.hybrid_matcher import HybridParallelMatcher, HybridSerialMatcher
from retrieval_service.app.core.matching.registry import MatcherRegistry
from retrieval_service.app.core.matching.vsm_matcher import VSMMatcher
from retrieval_service.app.core.search_engine import BM25SearchEngine, EmbeddingSearchEngine
from shared.index_store import JsonIndexStore


@pytest.fixture
def registry():
    store = JsonIndexStore(str(FIXTURE_DIR))
    vsm_cache = {}
    bm25 = BM25SearchEngine(store=store)
    embedding = EmbeddingSearchEngine(store=store)
    return MatcherRegistry(store, bm25, embedding, vsm_cache)


@pytest.fixture
def params():
    return MatchParams(k1=1.5, b=0.75, top_n_filter=2, k_rrf=60)


def test_vsm_cosine_ordering(registry, params):
    matcher = registry.get("vsm")
    query = QueryRepresentation(raw_text="hospital system", tokens=["hospital", "system"], mode="vsm")
    result = matcher.match(query, params)
    assert result.scores
    assert "doc1" in result.scores
    scores = list(result.scores.values())
    assert scores == sorted(scores, reverse=True)


def test_bm25_ordering(registry, params):
    matcher = registry.get("bm25")
    query = QueryRepresentation(raw_text="hospital system", tokens=["hospital", "system"], mode="bm25")
    result = matcher.match(query, params)
    assert result.scores
    assert result.scores["doc1"] > 0


def test_bm25_empty_tokens(registry, params):
    matcher = registry.get("bm25")
    query = QueryRepresentation(raw_text="", tokens=[], mode="bm25")
    result = matcher.match(query, params)
    assert result.scores == {}
    assert result.empty_reason == "no_lexical_overlap"


def test_embedding_ordering_with_mock(registry, params):
    matcher = registry.get("embedding")
    query = QueryRepresentation(raw_text="hospital", tokens=["hospital"], mode="embedding")

    mock_model = MagicMock()
    mock_model.encode.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch.object(registry.embedding_engine, "model", mock_model):
        with patch.object(registry.embedding_engine, "_lazy_load_model", lambda: None):
            result = matcher.match(query, params)

    assert result.scores
    assert result.scores.get("doc1", 0) >= result.scores.get("doc2", 0)


def test_hybrid_parallel_rrf(registry, params):
    matcher = registry.get("hybrid_parallel")
    query = QueryRepresentation(
        raw_text="hospital system security",
        tokens=["hospital", "system", "security"],
        mode="hybrid_parallel",
    )

    mock_model = MagicMock()
    mock_model.encode.return_value = [0.0, 0.0, 1.0, 0.0]

    with patch.object(registry.embedding_engine, "model", mock_model):
        with patch.object(registry.embedding_engine, "_lazy_load_model", lambda: None):
            result = matcher.match(query, params)

    assert result.scores
    scores = list(result.scores.values())
    assert scores == sorted(scores, reverse=True)


def test_hybrid_serial_respects_top_n_filter(registry, params):
    matcher = registry.get("hybrid_serial")
    query = QueryRepresentation(
        raw_text="hospital system",
        tokens=["hospital", "system"],
        mode="hybrid_serial",
    )
    params.top_n_filter = 1

    mock_model = MagicMock()
    mock_model.encode.return_value = [1.0, 0.0, 0.0, 0.0]

    with patch.object(registry.embedding_engine, "model", mock_model):
        with patch.object(registry.embedding_engine, "_lazy_load_model", lambda: None):
            result = matcher.match(query, params)

    assert len(result.scores) <= 1


def test_ranker_stable_tiebreak():
    scores = {"b": 1.0, "a": 1.0, "c": 0.5}
    ranked = Ranker.rank(scores, top_k=None)
    assert list(ranked.keys()) == ["a", "b", "c"]


def test_list_matchers_metadata():
    from retrieval_service.app.core.matching.registry import list_matchers

    matchers = list_matchers()
    assert len(matchers) == 5
    modes = {m["mode"] for m in matchers}
    assert modes == {"vsm", "bm25", "embedding", "hybrid_parallel", "hybrid_serial"}


def test_vsm_no_overlap(registry, params):
    matcher = VSMMatcher(JsonIndexStore(str(FIXTURE_DIR)), {})
    query = QueryRepresentation(raw_text="xyz", tokens=["xyz", "unknown"], mode="vsm")
    result = matcher.match(query, params)
    assert result.scores == {}
    assert result.empty_reason == "no_lexical_overlap"


def test_faiss_skipped_below_threshold(tmp_path):
    from shared.index_builder import IndexBuilder

    builder = IndexBuilder()
    builder.doc_embeddings = {"doc1": [1.0, 0.0, 0.0, 0.0]}
    info = builder._build_faiss_index(str(tmp_path))
    assert info["ann_backend"] == "none"
    assert info["vector_count"] == 1
