import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "query_refinement_service"))

from shared.query_suggestions import catalog_status, load_catalog, prefix_suggestions


SAMPLE_CATALOG = sorted(
    [
        "how much does a passport cost",
        "how to bake bread",
        "how to tie a tie",
        "what is machine learning",
        "where is paris",
    ]
)


class TestPrefixSuggestions:
    def test_prefix_match(self):
        results = prefix_suggestions(SAMPLE_CATALOG, "how to", limit=5)
        assert results == ["how to bake bread", "how to tie a tie"]

    def test_case_insensitive(self):
        results = prefix_suggestions(SAMPLE_CATALOG, "HOW TO", limit=5)
        assert len(results) == 2

    def test_limit_cap(self):
        results = prefix_suggestions(SAMPLE_CATALOG, "how", limit=1)
        assert len(results) == 1

    def test_no_match(self):
        assert prefix_suggestions(SAMPLE_CATALOG, "zzzzz", limit=5) == []

    def test_empty_prefix(self):
        assert prefix_suggestions(SAMPLE_CATALOG, "", limit=5) == []

    def test_empty_catalog(self):
        assert prefix_suggestions([], "how", limit=5) == []


class TestLoadCatalog:
    def test_load_from_file(self, tmp_path):
        catalog_path = tmp_path / "query_suggestions.json"
        catalog_path.write_text(
            json.dumps({"queries": SAMPLE_CATALOG}),
            encoding="utf-8",
        )
        assert load_catalog(catalog_path) == SAMPLE_CATALOG

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_catalog(tmp_path / "missing.json") == []

    def test_catalog_status(self, tmp_path):
        catalog_path = tmp_path / "query_suggestions.json"
        catalog_path.write_text(
            json.dumps({"queries": SAMPLE_CATALOG}),
            encoding="utf-8",
        )
        status = catalog_status(catalog_path)
        assert status == {"loaded": True, "count": len(SAMPLE_CATALOG)}


class TestSuggestEndpoint:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        catalog_path = tmp_path / "query_suggestions.json"
        catalog_path.write_text(
            json.dumps({"queries": SAMPLE_CATALOG}),
            encoding="utf-8",
        )
        monkeypatch.setenv("IR_QUERY_SUGGESTIONS", str(catalog_path))
        monkeypatch.setenv("IR_SUGGEST_MIN_PREFIX_LEN", "2")

        import shared.ir_config as ir_config
        import shared.query_suggestions as query_suggestions_module
        import query_refinement_service.app.main as refinement_main

        monkeypatch.setattr(ir_config, "QUERY_SUGGESTIONS_PATH", str(catalog_path))
        monkeypatch.setattr(refinement_main, "QUERY_SUGGESTIONS_PATH", str(catalog_path))
        query_suggestions_module._cache = (None, [])

        from query_refinement_service.app.main import app

        return TestClient(app)

    def test_suggest_returns_matches(self, client):
        response = client.get("/suggest", params={"q": "how to", "limit": 5})
        assert response.status_code == 200
        payload = response.json()
        assert payload["query_prefix"] == "how to"
        assert "how to tie a tie" in payload["suggestions"]

    def test_short_prefix_returns_empty(self, client):
        response = client.get("/suggest", params={"q": "h", "limit": 5})
        assert response.status_code == 200
        assert response.json()["suggestions"] == []

    def test_health_includes_suggestion_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["suggestions_index_loaded"] is True
        assert payload["suggestions_count"] == len(SAMPLE_CATALOG)
