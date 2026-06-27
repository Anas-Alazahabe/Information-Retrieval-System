"""Tests for offline cluster precompute."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering_service.app.core.loader import cluster_artifacts_ready, load_cluster_manifest
from clustering_service.app.core.precompute import run_precompute
from shared.ir_config import CLUSTER_ARTIFACT_FILES


@pytest.fixture
def cluster_index_dir(tmp_path):
    fixture_path = ROOT / "tests" / "fixtures" / "cluster_index" / "embeddings_index.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        embeddings = json.load(f)
    with open(tmp_path / "embeddings_index.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
    return str(tmp_path)


class TestClusteringPrecompute:
    def test_run_precompute_creates_all_artifacts(self, cluster_index_dir):
        manifest = run_precompute(
            index_dir=cluster_index_dir,
            max_k=5,
            viz_max_points=10,
        )
        assert cluster_artifacts_ready(cluster_index_dir)
        for filename in CLUSTER_ARTIFACT_FILES:
            assert os.path.isfile(os.path.join(cluster_index_dir, filename))

        assert manifest["total_docs"] == 15
        assert 2 <= manifest["n_clusters"] <= 5
        assert manifest["viz_sample_size"] <= 10

    def test_labels_match_doc_count(self, cluster_index_dir):
        run_precompute(index_dir=cluster_index_dir, max_k=4, viz_max_points=10)
        with open(os.path.join(cluster_index_dir, "cluster_doc_ids.json"), "r") as f:
            doc_ids = json.load(f)
        labels = np.load(os.path.join(cluster_index_dir, "all_labels.npy"))
        assert len(doc_ids) == len(labels) == 15

    def test_manifest_fields(self, cluster_index_dir):
        run_precompute(index_dir=cluster_index_dir, max_k=4, viz_max_points=8)
        manifest = load_cluster_manifest(cluster_index_dir)
        assert "timestamp" in manifest
        assert manifest["n_clusters"] >= 2
