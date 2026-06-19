import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "query_refinement_service"))

from preprocessing_service.app.core.cleaner import TextCleaner
from query_refinement_service.app.core.refiner import (
    QueryRefiner,
    apply_query_preprocess,
    is_question_query,
)
from query_refinement_service.app.main import app as refinement_app


@pytest.fixture(scope="module")
def cleaner():
    return TextCleaner()


@pytest.fixture(scope="module")
def refiner():
    return QueryRefiner()


@pytest.fixture(scope="module")
def refinement_client():
    return TestClient(refinement_app)


class TestQuestionDetection:
    def test_wh_question(self):
        assert is_question_query("what is machine learning") is True

    def test_trailing_question_mark(self):
        assert is_question_query("capital of france?") is True

    def test_non_question(self):
        assert is_question_query("hospital system") is False


class TestTextCleanerPreserveWhWords:
    def test_wh_question_preserves_wh_word(self, cleaner):
        tokens = cleaner.process(
            "what is machine learning",
            use_lemmatization=True,
            remove_stop=True,
            preserve_wh_words=True,
        )
        assert "what" in tokens
        assert "is" not in tokens
        assert "machine" in tokens
        assert "learning" in tokens

    def test_wh_question_default_removes_wh_word(self, cleaner):
        tokens = cleaner.process(
            "what is machine learning",
            use_lemmatization=True,
            remove_stop=True,
            preserve_wh_words=False,
        )
        assert "what" not in tokens
        assert "is" not in tokens

    def test_non_question_unaffected_by_preserve_flag(self, cleaner):
        without = cleaner.process(
            "hospital system",
            use_lemmatization=True,
            remove_stop=True,
            preserve_wh_words=False,
        )
        with_flag = cleaner.process(
            "hospital system",
            use_lemmatization=True,
            remove_stop=True,
            preserve_wh_words=True,
        )
        assert without == with_flag


class TestQueryRefiner:
    def test_pass_through_no_techniques(self, refiner):
        result = refiner.refine("  hospital system  ", enabled_techniques=[])
        assert result["refined_query"] == "hospital system"
        assert result["techniques_applied"] == []
        assert result["preprocess_hints"] == {}
        assert "No refinement techniques enabled" in result["explanation"]

    def test_query_preprocess_on_question(self, refiner):
        result = refiner.refine(
            "what is machine learning",
            enabled_techniques=["query_preprocess"],
        )
        assert result["techniques_applied"] == ["query_preprocess"]
        assert result["preprocess_hints"].get("preserve_wh_words") is True

    def test_query_preprocess_on_non_question(self, refiner):
        result = refiner.refine(
            "hospital system",
            enabled_techniques=["query_preprocess"],
        )
        assert result["preprocess_hints"] == {}

    def test_apply_query_preprocess_returns_hints(self):
        _, hints, explanation = apply_query_preprocess("how does bm25 work?")
        assert hints == {"preserve_wh_words": True}
        assert "Question-style" in explanation


class TestRefinementServiceApi:
    def test_health(self, refinement_client):
        response = refinement_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_refine_pass_through(self, refinement_client):
        response = refinement_client.post(
            "/refine",
            json={"raw_query": "hospital system", "enabled_techniques": []},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["refined_query"] == "hospital system"
        assert body["techniques_applied"] == []

    def test_refine_query_preprocess(self, refinement_client):
        response = refinement_client.post(
            "/refine",
            json={
                "raw_query": "what is information retrieval",
                "enabled_techniques": ["query_preprocess"],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["preprocess_hints"].get("preserve_wh_words") is True
        assert "query_preprocess" in body["techniques_applied"]

    def test_refine_empty_query(self, refinement_client):
        response = refinement_client.post(
            "/refine",
            json={"raw_query": "   ", "enabled_techniques": []},
        )
        assert response.status_code == 400
