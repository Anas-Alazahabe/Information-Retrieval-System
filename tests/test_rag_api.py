"""FastAPI tests for rag_service with mocked Gemini."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAG_SERVICE = ROOT / "rag_service"
if str(RAG_SERVICE) not in sys.path:
    sys.path.insert(0, str(RAG_SERVICE))


@pytest.fixture
def rag_client():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
        import importlib

        import shared.ir_config as ir_config

        importlib.reload(ir_config)

        import rag_service.app.main as rag_main

        importlib.reload(rag_main)
        yield TestClient(rag_main.app)


def test_health_reports_gemini_configured(rag_client):
    response = rag_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "rag_service"
    assert data["gemini_configured"] is True


def test_generate_success(rag_client):
    context_patch = patch(
        "rag_service.app.main.build_context",
        return_value={
            "context_text": "[DOC d1 score=9.00]\nPassage one.",
            "context_doc_ids": ["d1"],
            "missing_doc_ids": [],
            "citations": [
                {
                    "doc_id": "d1",
                    "snippet": "Passage one.",
                    "retrieval_score": 9.0,
                }
            ],
            "context_chars": 20,
        },
    )
    gemini_patch = patch(
        "rag_service.app.main.generate_answer",
        return_value="Answer with citation [DOC d1].",
    )

    with context_patch, gemini_patch:
        response = rag_client.post(
            "/generate",
            json={
                "query": "test query",
                "results": {"d1": 9.0},
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Answer" in data["answer"]
    assert data["context_doc_ids"] == ["d1"]


def test_generate_no_passage_text_returns_422(rag_client):
    with patch(
        "rag_service.app.main.build_context",
        return_value={
            "context_text": "",
            "context_doc_ids": [],
            "missing_doc_ids": ["d1"],
            "citations": [],
            "context_chars": 0,
        },
    ):
        response = rag_client.post(
            "/generate",
            json={"query": "q", "results": {"d1": 1.0}},
        )

    assert response.status_code == 422


def test_generate_empty_results_returns_400(rag_client):
    response = rag_client.post("/generate", json={"query": "q", "results": {}})
    assert response.status_code == 400
