import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.ir_config import CLUSTER_ARTIFACT_FILES, INDEX_DIR


def _artifact_path(index_dir: str, filename: str) -> str:
    return os.path.join(index_dir, filename)


def missing_cluster_artifacts(index_dir: str = INDEX_DIR) -> List[str]:
    """Return list of missing cluster artifact filenames."""
    missing = []
    for filename in CLUSTER_ARTIFACT_FILES:
        if not os.path.isfile(_artifact_path(index_dir, filename)):
            missing.append(filename)
    return missing


def cluster_artifacts_ready(index_dir: str = INDEX_DIR) -> bool:
    return len(missing_cluster_artifacts(index_dir)) == 0


def load_cluster_manifest(index_dir: str = INDEX_DIR) -> Dict[str, Any]:
    path = _artifact_path(index_dir, "cluster_manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index_manifest(index_dir: str = INDEX_DIR) -> Dict[str, Any]:
    path = _artifact_path(index_dir, "index_manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_doc_ids(index_dir: str = INDEX_DIR) -> List[str]:
    path = _artifact_path(index_dir, "cluster_doc_ids.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels(index_dir: str = INDEX_DIR) -> np.ndarray:
    path = _artifact_path(index_dir, "all_labels.npy")
    return np.load(path)


def load_cluster_model(index_dir: str = INDEX_DIR):
    path = _artifact_path(index_dir, "cluster_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tsne_data(index_dir: str = INDEX_DIR) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load cached t-SNE coordinates, labels for subsample, and doc IDs."""
    coords = np.load(_artifact_path(index_dir, "tsne_coords.npy"))
    labels = np.load(_artifact_path(index_dir, "tsne_labels.npy"))
    with open(_artifact_path(index_dir, "tsne_doc_ids.json"), "r", encoding="utf-8") as f:
        doc_ids = json.load(f)
    return coords, labels, doc_ids


def build_cluster_meta(index_dir: str = INDEX_DIR, sample_per_cluster: int = 5) -> Dict[str, Any]:
    manifest = load_cluster_manifest(index_dir)
    doc_ids = load_doc_ids(index_dir)
    labels = load_labels(index_dir)
    index_manifest = load_index_manifest(index_dir)

    clusters: Dict[int, List[str]] = {}
    for doc_id, label in zip(doc_ids, labels):
        clusters.setdefault(int(label), []).append(doc_id)

    cluster_summaries = []
    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "sample_doc_ids": members[:sample_per_cluster],
            }
        )

    return {
        "document_count": manifest.get("total_docs", len(doc_ids)),
        "n_clusters": manifest.get("n_clusters", len(cluster_summaries)),
        "viz_sample_size": manifest.get("viz_sample_size", 0),
        "embedding_model": manifest.get(
            "embedding_model", index_manifest.get("embedding_model")
        ),
        "clusters": cluster_summaries,
    }


def get_health_info(index_dir: str = INDEX_DIR) -> Dict[str, Any]:
    missing = missing_cluster_artifacts(index_dir)
    ready = len(missing) == 0
    info: Dict[str, Any] = {
        "cluster_artifacts_ready": ready,
        "missing_artifacts": missing,
        "document_count": None,
        "n_clusters": None,
        "embedding_model": None,
    }
    if ready:
        manifest = load_cluster_manifest(index_dir)
        info["document_count"] = manifest.get("total_docs")
        info["n_clusters"] = manifest.get("n_clusters")
        info["embedding_model"] = manifest.get("embedding_model")
    else:
        id_map_path = os.path.join(index_dir, "embeddings_id_map.json")
        if os.path.isfile(id_map_path):
            with open(id_map_path, "r", encoding="utf-8") as f:
                info["document_count"] = len(json.load(f))
        elif os.path.isfile(_artifact_path(index_dir, "embeddings_index.json")):
            with open(_artifact_path(index_dir, "embeddings_index.json"), "r", encoding="utf-8") as f:
                info["document_count"] = len(json.load(f))
    return info
