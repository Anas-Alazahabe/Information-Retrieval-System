"""API tests for personalization service with mocked persistence."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from personalization_service.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestPersonalizationApi:
    @patch("personalization_service.app.main.get_interest_terms")
    @patch("personalization_service.app.main.rerank_results")
    def test_rerank_endpoint(self, mock_rerank, mock_terms, client):
        mock_terms.return_value = {"python": 2.0}
        mock_rerank.return_value = (
            {"doc2": 0.9, "doc1": 0.5},
            {
                "personalization_applied": True,
                "alpha": 0.7,
                "profile_terms_used": ["python"],
                "boosted_docs": [],
                "explanation": "ok",
                "missing_doc_count": 0,
            },
        )

        response = client.post(
            "/personalize/rerank",
            json={
                "user_id": "demo_tech",
                "query_text": "learn programming",
                "results": {"doc1": 0.8, "doc2": 0.6},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["personalization_applied"] is True
        assert payload["results"]["doc2"] == 0.9

    @patch("personalization_service.app.main.check_db_connection")
    def test_health_degraded_without_db(self, mock_db, client):
        mock_db.return_value = {"connected": False, "error": "connection refused"}
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    @patch("personalization_service.app.main.upsert_interest_terms")
    @patch("personalization_service.app.main.log_query_event")
    @patch("personalization_service.app.main.terms_from_query")
    def test_query_event(self, mock_terms, mock_log, mock_upsert, client):
        mock_terms.return_value = {"health": 1.0}
        response = client.post(
            "/events/query",
            json={"user_id": "demo_health", "query_text": "health care"},
        )
        assert response.status_code == 200
        assert response.json()["terms_added"] == ["health"]
        mock_log.assert_called_once()
        mock_upsert.assert_called_once()
