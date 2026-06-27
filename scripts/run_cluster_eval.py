"""Document clustering quality evaluation (intrinsic metrics)."""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

from shared.ir_config import CLUSTERING_URL, INDEX_DIR

CLUSTER_MANIFEST = "cluster_manifest.json"
ALL_LABELS = "all_labels.npy"
TSNE_LABELS = "tsne_labels.npy"


def _load_cluster_artifacts(index_dir: str) -> tuple[dict, np.ndarray, Optional[np.ndarray]]:
    manifest_path = os.path.join(index_dir, CLUSTER_MANIFEST)
    labels_path = os.path.join(index_dir, ALL_LABELS)
    tsne_labels_path = os.path.join(index_dir, TSNE_LABELS)

    if not os.path.isfile(manifest_path) or not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"Cluster artifacts missing in {index_dir}. Run scripts/run_cluster_precompute.py first."
        )

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    labels = np.load(labels_path)
    tsne_labels = np.load(tsne_labels_path) if os.path.isfile(tsne_labels_path) else None
    return manifest, labels, tsne_labels


def _cluster_size_stats(labels: np.ndarray) -> dict:
    counts = Counter(labels.tolist())
    sizes = sorted(counts.values())
    return {
        "n_clusters": len(counts),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": round(float(np.mean(sizes)), 2),
        "std_size": round(float(np.std(sizes)), 2),
        "sizes": dict(sorted((int(k), v) for k, v in counts.items())),
    }


def _compute_silhouette(index_dir: str, labels: np.ndarray, sample_size: int = 5000) -> Optional[float]:
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None

    from clustering_service.app.core.precompute import _load_embeddings

    try:
        doc_ids, embeddings = _load_embeddings(index_dir)
    except Exception:
        return None

    if len(doc_ids) != len(labels):
        return None

    n = len(labels)
    if n > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_size, replace=False)
        embeddings = embeddings[idx]
        labels_sample = labels[idx]
    else:
        labels_sample = labels

    if len(set(labels_sample.tolist())) < 2:
        return None

    return round(float(silhouette_score(embeddings, labels_sample)), 4)


def run_cluster_evaluation(
    *,
    index_dir: str = INDEX_DIR,
    clustering_url: str = CLUSTERING_URL,
) -> Dict:
    manifest, labels, tsne_labels = _load_cluster_artifacts(index_dir)
    size_stats = _cluster_size_stats(labels)
    silhouette = _compute_silhouette(index_dir, labels)

    tsne_plot_path = os.path.join(index_dir, "cluster_comparison.png")
    try:
        import requests

        response = requests.get(
            f"{clustering_url.rstrip('/')}/cluster/comparison",
            timeout=60,
        )
        if response.status_code == 200:
            with open(tsne_plot_path, "wb") as handle:
                handle.write(response.content)
    except Exception:
        pass

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "index_dir": index_dir,
        "manifest": manifest,
        "cluster_stats": size_stats,
        "silhouette_score": silhouette,
        "tsne_viz_sample_size": int(len(tsne_labels)) if tsne_labels is not None else None,
        "tsne_plot_path": tsne_plot_path if os.path.isfile(tsne_plot_path) else None,
        "limitations": [
            "Clustering is visualization-only and does not affect retrieval rankings.",
            "Silhouette computed on embedding subsample when index is large.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering quality evaluation")
    parser.add_argument("--scale", default="full", choices=["dev", "preval", "full"])
    parser.add_argument("--index-dir", default=INDEX_DIR)
    parser.add_argument("--clustering-url", default=CLUSTERING_URL)
    parser.add_argument("--output-dir", default=str(ROOT / "evaluation_results"))
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("Prerequisites: cluster precompute artifacts in index_dir.")

    report = run_cluster_evaluation(
        index_dir=args.index_dir,
        clustering_url=args.clustering_url,
    )
    report["scale"] = args.scale

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cluster_eval_{args.scale}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved {output_path}")
    print(f"Clusters: {report['cluster_stats']['n_clusters']}, silhouette: {report['silhouette_score']}")


if __name__ == "__main__":
    main()
