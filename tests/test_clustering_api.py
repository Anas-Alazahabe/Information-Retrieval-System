"""API tests for clustering service."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering_service.app.core.precompute import run_precompute


@pytest.fixture
def cluster_client(tmp_path):
    fixture_path = ROOT / "tests" / "fixtures" / "cluster_index" / "embeddings_index.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        embeddings = json.load(f)
    index_dir = str(tmp_path)
    with open(os.path.join(index_dir, "embeddings_index.json"), "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
    run_precompute(index_dir=index_dir, max_k=4, viz_max_points=10)

    with patch("clustering_service.app.main.INDEX_DIR", index_dir), patch(
        "clustering_service.app.core.loader.INDEX_DIR", index_dir
    ):
        from clustering_service.app.main import app

        yield TestClient(app), index_dir


class TestClusteringApi:
    def test_health_ready(self, cluster_client):
        client, _ = cluster_client
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["service"] == "clustering_service"
        assert payload["cluster_artifacts_ready"] is True
        assert payload["document_count"] == 15
        assert payload["n_clusters"] >= 2

    def test_cluster_meta(self, cluster_client):
        client, _ = cluster_client
        response = client.get("/cluster/meta")
        assert response.status_code == 200
        payload = response.json()
        assert payload["document_count"] == 15
        assert len(payload["clusters"]) >= 2
        assert all("sample_doc_ids" in c for c in payload["clusters"])

    def test_cluster_comparison_png(self, cluster_client):
        client, _ = cluster_client
        response = client.get("/cluster/comparison")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 1000

    def test_health_degraded_without_artifacts(self, tmp_path):
        with patch("clustering_service.app.main.INDEX_DIR", str(tmp_path)), patch(
            "clustering_service.app.core.loader.INDEX_DIR", str(tmp_path)
        ):
            from clustering_service.app.main import app

            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "degraded"
            assert payload["cluster_artifacts_ready"] is False

    def test_meta_503_without_artifacts(self, tmp_path):
        with patch("clustering_service.app.main.INDEX_DIR", str(tmp_path)), patch(
            "clustering_service.app.core.loader.INDEX_DIR", str(tmp_path)
        ):
            from clustering_service.app.main import app

            client = TestClient(app)
            response = client.get("/cluster/meta")
            assert response.status_code == 503
