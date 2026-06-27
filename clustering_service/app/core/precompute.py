"""Offline K-Means clustering and t-SNE visualization cache."""

import json
import os
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE

from shared.ir_config import (
    CLUSTER_MINIBATCH_THRESHOLD,
    CLUSTER_NUM_CLUSTERS_MAX,
    CLUSTER_VIZ_MAX_POINTS,
    INDEX_DIR,
)


def _load_embeddings_from_faiss(index_dir: str) -> Optional[Tuple[List[str], np.ndarray]]:
    faiss_path = os.path.join(index_dir, "embeddings.faiss")
    id_map_path = os.path.join(index_dir, "embeddings_id_map.json")
    if not (os.path.isfile(faiss_path) and os.path.isfile(id_map_path)):
        return None

    try:
        import faiss
    except ImportError:
        return None

    with open(id_map_path, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)

    index = faiss.read_index(faiss_path)
    if index.ntotal != len(doc_ids):
        print(
            f"Warning: FAISS vector count ({index.ntotal}) != id map ({len(doc_ids)}); "
            "falling back to embeddings_index.json"
        )
        return None

    vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)
    return doc_ids, vectors.astype(np.float64)


def _load_embeddings_from_json(index_dir: str) -> Tuple[List[str], np.ndarray]:
    json_path = os.path.join(index_dir, "embeddings_index.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"embeddings_index.json not found in {index_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        data: Dict[str, List[float]] = json.load(f)

    if not data:
        raise ValueError("embeddings_index.json is empty")

    doc_ids = list(data.keys())
    embeddings = np.array([data[doc_id] for doc_id in doc_ids], dtype=np.float64)
    return doc_ids, embeddings


def _load_embeddings(index_dir: str) -> Tuple[List[str], np.ndarray]:
    faiss_data = _load_embeddings_from_faiss(index_dir)
    if faiss_data is not None:
        print(f"Loaded {faiss_data[1].shape[0]} embeddings from FAISS index")
        return faiss_data
    print("Loading embeddings from embeddings_index.json...")
    return _load_embeddings_from_json(index_dir)


def _choose_num_clusters(n_samples: int, max_k: int) -> int:
    return min(max_k, max(2, n_samples))


def _fit_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
):
    n_samples = embeddings.shape[0]
    if n_samples > CLUSTER_MINIBATCH_THRESHOLD:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=min(4096, n_samples),
            n_init=3,
        )
    else:
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    model.fit(embeddings)
    return model


def _stratified_subsample(
    doc_ids: List[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    max_points: int,
    random_state: int = 42,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    n_samples = len(doc_ids)
    if n_samples <= max_points:
        return doc_ids, labels, embeddings

    rng = np.random.default_rng(random_state)
    unique_labels = np.unique(labels)
    per_cluster = max(1, max_points // len(unique_labels))
    selected_indices: List[int] = []

    for cluster_id in unique_labels:
        cluster_indices = np.where(labels == cluster_id)[0]
        take = min(per_cluster, len(cluster_indices))
        chosen = rng.choice(cluster_indices, size=take, replace=False)
        selected_indices.extend(chosen.tolist())

    if len(selected_indices) < max_points:
        remaining = sorted(set(range(n_samples)) - set(selected_indices))
        extra = min(max_points - len(selected_indices), len(remaining))
        if extra > 0:
            selected_indices.extend(
                rng.choice(remaining, size=extra, replace=False).tolist()
            )

    selected_indices = selected_indices[:max_points]
    idx = np.array(sorted(selected_indices))
    sub_doc_ids = [doc_ids[i] for i in idx]
    sub_labels = labels[idx]
    sub_embeddings = embeddings[idx]
    return sub_doc_ids, sub_labels, sub_embeddings


def _compute_tsne(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    n_samples = embeddings.shape[0]
    perp = min(30, max(1, n_samples - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def run_precompute(
    index_dir: str = INDEX_DIR,
    max_k: Optional[int] = None,
    viz_max_points: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """Run K-Means clustering and cache visualization artifacts."""
    max_k = max_k if max_k is not None else CLUSTER_NUM_CLUSTERS_MAX
    viz_max_points = viz_max_points if viz_max_points is not None else CLUSTER_VIZ_MAX_POINTS

    os.makedirs(index_dir, exist_ok=True)
    doc_ids, embeddings = _load_embeddings(index_dir)
    n_samples = len(doc_ids)
    n_clusters = _choose_num_clusters(n_samples, max_k)

    print(f"Clustering {n_samples} documents into {n_clusters} clusters...")
    model = _fit_kmeans(embeddings, n_clusters, random_state=random_state)
    labels = model.labels_

    viz_doc_ids, viz_labels, viz_embeddings = _stratified_subsample(
        doc_ids, labels, embeddings, viz_max_points, random_state=random_state
    )
    print(f"Computing t-SNE for {len(viz_doc_ids)} visualization points...")
    tsne_coords = _compute_tsne(viz_embeddings, random_state=random_state)

    index_manifest_path = os.path.join(index_dir, "index_manifest.json")
    embedding_model = None
    if os.path.isfile(index_manifest_path):
        with open(index_manifest_path, "r", encoding="utf-8") as f:
            embedding_model = json.load(f).get("embedding_model")

    cluster_manifest = {
        "total_docs": n_samples,
        "n_clusters": n_clusters,
        "viz_sample_size": len(viz_doc_ids),
        "embedding_model": embedding_model,
        "algorithm": type(model).__name__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(os.path.join(index_dir, "cluster_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    np.save(os.path.join(index_dir, "all_labels.npy"), labels)
    with open(os.path.join(index_dir, "cluster_doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    with open(os.path.join(index_dir, "cluster_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_manifest, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(index_dir, "tsne_coords.npy"), tsne_coords)
    np.save(os.path.join(index_dir, "tsne_labels.npy"), viz_labels)
    with open(os.path.join(index_dir, "tsne_doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(viz_doc_ids, f, ensure_ascii=False)

    print(f"Cluster artifacts saved to {index_dir}")
    return cluster_manifest
