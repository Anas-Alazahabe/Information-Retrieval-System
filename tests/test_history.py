import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "query_refinement_service"))

from query_refinement_service.app.core.history import merge_history_terms
from query_refinement_service.app.core.refiner import QueryRefiner
from query_refinement_service.app.main import app as refinement_app


@pytest.fixture(scope="module")
def refiner():
    return QueryRefiner()


@pytest.fixture(scope="module")
def refinement_client():
    return TestClient(refinement_app)


class TestMergeHistoryTerms:
    def test_decay_prefers_recent_query(self):
        terms, explanation = merge_history_terms(
            current_query="neural networks",
            current_tokens=["neural", "networks"],
            previous_queries=["machine learning basics", "deep learning tutorial"],
            max_terms=3,
        )
        assert "learning" in terms
        assert "deep" in terms or "tutorial" in terms
        assert "History added" in explanation

    def test_skips_terms_already_in_current_query(self):
        terms, _ = merge_history_terms(
            current_query="machine learning",
            current_tokens=["machine", "learning"],
            previous_queries=["machine learning basics"],
            max_terms=5,
        )
        assert "machine" not in terms
        assert "learning" not in terms
        assert "basics" in terms

    def test_respects_max_terms_cap(self):
        terms, _ = merge_history_terms(
            current_query="alpha",
            current_tokens=["alpha"],
            previous_queries=["beta gamma delta epsilon zeta eta"],
            max_terms=2,
        )
        assert len(terms) <= 2

    def test_empty_history(self):
        terms, explanation = merge_history_terms(
            current_query="test query",
            current_tokens=["test", "query"],
            previous_queries=[],
        )
        assert terms == []
        assert "No session history" in explanation

    def test_excludes_current_query_from_history(self):
        terms, explanation = merge_history_terms(
            current_query="machine learning",
            current_tokens=["machine", "learning"],
            previous_queries=["machine learning", "deep learning"],
        )
        assert "deep" in terms
        assert "History added" in explanation


class TestRefinerHistoryIntegration:
    def test_history_adds_terms(self, refiner):
        result = refiner.refine(
            raw_query="neural networks",
            enabled_techniques=["history"],
            previous_queries=["machine learning"],
        )
        assert "history" in result["techniques_applied"]
        assert "machine" in result["expanded_terms"] or "learning" in result["expanded_terms"]
        assert "History added" in result["explanation"]

    def test_history_disabled_without_technique(self, refiner):
        result = refiner.refine(
            raw_query="neural networks",
            enabled_techniques=["query_preprocess"],
            previous_queries=["machine learning"],
        )
        assert "history" not in result["techniques_applied"]


class TestHistoryApi:
    def test_refine_with_history(self, refinement_client):
        response = refinement_client.post(
            "/refine",
            json={
                "raw_query": "neural networks",
                "enabled_techniques": ["history"],
                "previous_queries": ["machine learning"],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert "history" in payload["techniques_applied"]
        assert payload["refined_query"] != payload["raw_query"]
