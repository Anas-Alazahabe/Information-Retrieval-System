"""Unit tests for personalization re-ranking."""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from personalization_service.app.core.reranker import rerank_results


class TestRerankResults:
    def test_empty_results(self):
        reranked, meta = rerank_results({}, {"health": 2.0})
        assert reranked == {}
        assert meta["personalization_applied"] is False

    def test_empty_profile_returns_original_order(self):
        results = {"doc_a": 0.9, "doc_b": 0.5}
        reranked, meta = rerank_results(results, {})
        assert list(reranked.keys()) == ["doc_a", "doc_b"]
        assert meta["personalization_applied"] is False

    @patch("personalization_service.app.core.reranker.fetch_document_texts")
    def test_profile_boost_changes_ranking(self, mock_fetch):
        mock_fetch.return_value = {
            "doc_a": "general article about travel",
            "doc_b": "diabetes symptoms and insulin treatment guide",
        }
        results = {"doc_a": 0.95, "doc_b": 0.70}
        profile = {"diabetes": 3.0, "insulin": 2.0}

        reranked, meta = rerank_results(
            results,
            profile,
            alpha=0.3,
            query_text="what is travel",
        )

        assert meta["personalization_applied"] is True
        assert list(reranked.keys())[0] == "doc_b"
        assert meta["boosted_docs"]

    @patch("personalization_service.app.core.reranker.fetch_document_texts")
    def test_missing_doc_gets_zero_profile_boost(self, mock_fetch):
        mock_fetch.return_value = {"doc_a": "diabetes health medical"}
        results = {"doc_a": 0.8, "doc_missing": 0.9}
        profile = {"diabetes": 2.0}

        reranked, meta = rerank_results(results, profile, alpha=0.7)
        assert "doc_missing" in reranked
        assert meta["missing_doc_count"] == 1
