import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "query_refinement_service"))

from app.core.prf import (
    _is_valid_feedback_term,
    collect_doc_term_weights,
    prf_rm3,
)
from app.core.refiner import QueryRefiner

SAMPLE_BM25_INDEX = {
    "hospital": {
        "doc_a": {"tf": 2, "doc_len": 10},
        "doc_b": {"tf": 1, "doc_len": 8},
    },
    "system": {
        "doc_a": {"tf": 1, "doc_len": 10},
    },
    "clinical": {
        "doc_a": {"tf": 3, "doc_len": 10},
    },
    "medicine": {
        "doc_b": {"tf": 2, "doc_len": 8},
    },
}

SAMPLE_BM25_INDEX_WITH_NOISE = {
    **SAMPLE_BM25_INDEX,
    "itâ€™s": {"doc_a": {"tf": 5, "doc_len": 10}},
    "café": {"doc_a": {"tf": 4, "doc_len": 10}},
    "noise!": {"doc_a": {"tf": 3, "doc_len": 10}},
}

SAMPLE_METADATA = {
    "doc_lengths": {"doc_a": 10, "doc_b": 8},
}


class TestCollectDocTermWeights:
    def test_returns_terms_for_doc(self):
        weights = collect_doc_term_weights("doc_a", SAMPLE_BM25_INDEX)
        assert weights == {"hospital": 2, "system": 1, "clinical": 3}

    def test_unknown_doc_returns_empty(self):
        assert collect_doc_term_weights("missing", SAMPLE_BM25_INDEX) == {}


class TestIsValidFeedbackTerm:
    def test_valid_terms(self):
        assert _is_valid_feedback_term("clinical") is True
        assert _is_valid_feedback_term("medicine") is True

    def test_filters_mojibake(self):
        assert _is_valid_feedback_term("itâ€™s") is False

    def test_filters_non_ascii(self):
        assert _is_valid_feedback_term("café") is False

    def test_filters_punctuation(self):
        assert _is_valid_feedback_term("noise!") is False


class TestPrfRm3:
    @patch("app.core.prf.JsonIndexStore")
    @patch("app.core.prf._first_pass_retrieval")
    def test_returns_feedback_terms(self, mock_first_pass, mock_store_cls):
        mock_first_pass.return_value = {"doc_a": 1.5, "doc_b": 0.8}
        mock_store = MagicMock()
        mock_store.load_bm25.return_value = SAMPLE_BM25_INDEX
        mock_store.load_metadata.return_value = SAMPLE_METADATA
        mock_store_cls.return_value = mock_store

        terms, explanation = prf_rm3(["health"], top_m_terms=3)

        assert terms
        assert "PRF from top-2 BM25 docs" in explanation
        assert "doc_a" in explanation

    @patch("app.core.prf._first_pass_retrieval")
    def test_empty_first_pass(self, mock_first_pass):
        mock_first_pass.return_value = {}
        terms, explanation = prf_rm3(["health"])
        assert terms == []
        assert "no first-pass results" in explanation

    @patch("app.core.prf._first_pass_retrieval")
    def test_retrieval_unavailable(self, mock_first_pass):
        mock_first_pass.side_effect = ConnectionError("offline")
        terms, explanation = prf_rm3(["health"])
        assert terms == []
        assert "retrieval unavailable" in explanation

    @patch("app.core.prf.JsonIndexStore")
    @patch("app.core.prf._first_pass_retrieval")
    def test_filters_invalid_index_terms(self, mock_first_pass, mock_store_cls):
        mock_first_pass.return_value = {"doc_a": 1.5}
        mock_store = MagicMock()
        mock_store.load_bm25.return_value = SAMPLE_BM25_INDEX_WITH_NOISE
        mock_store.load_metadata.return_value = SAMPLE_METADATA
        mock_store_cls.return_value = mock_store

        terms, explanation = prf_rm3(["health"], top_m_terms=10)

        assert "itâ€™s" not in terms
        assert "café" not in terms
        assert "clinical" in terms or "medicine" in terms
        assert "filtered" in explanation

    @patch("app.core.prf.JsonIndexStore")
    @patch("app.core.prf._first_pass_retrieval")
    def test_short_query_caps_top_m(self, mock_first_pass, mock_store_cls):
        mock_first_pass.return_value = {"doc_a": 1.5, "doc_b": 0.8}
        mock_store = MagicMock()
        mock_store.load_bm25.return_value = SAMPLE_BM25_INDEX
        mock_store.load_metadata.return_value = SAMPLE_METADATA
        mock_store_cls.return_value = mock_store

        terms, explanation = prf_rm3(["car"], top_m_terms=15)

        assert len(terms) <= 5
        assert "short-query cap" in explanation


class TestRefinerPrfIntegration:
    @patch("app.core.refiner.prf_rm3")
    @patch("app.core.refiner.QueryRefiner._fetch_query_tokens")
    def test_prf_technique_applied(self, mock_tokens, mock_prf):
        mock_tokens.return_value = ["hospital"]
        mock_prf.return_value = (["clinical"], "PRF added clinical.")
        refiner = QueryRefiner()
        result = refiner.refine("hospital", enabled_techniques=["prf"])
        assert "prf" in result["techniques_applied"]
        assert "clinical" in result["expanded_terms"]
        assert "clinical" in result["refined_query"]

    @patch("app.core.refiner.prf_rm3")
    @patch("app.core.refiner.expand_synonyms")
    @patch("app.core.refiner.QueryRefiner._fetch_query_tokens")
    def test_combined_techniques(self, mock_tokens, mock_synonyms, mock_prf):
        mock_tokens.return_value = ["what", "information"]
        mock_synonyms.return_value = ["info"]
        mock_prf.return_value = (["data"], "PRF added data.")
        refiner = QueryRefiner()
        result = refiner.refine(
            "what is information",
            enabled_techniques=["query_preprocess", "synonyms", "prf"],
        )
        assert result["techniques_applied"] == [
            "query_preprocess",
            "synonyms",
            "prf",
        ]
        assert result["preprocess_hints"].get("preserve_wh_words") is True
