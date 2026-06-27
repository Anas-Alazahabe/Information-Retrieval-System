import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))

from app.core.loader import (
    build_cluster_meta,
    cluster_artifacts_ready,
    get_health_info,
    load_cluster_manifest,
    load_tsne_data,
)
from app.core.visualize import render_cluster_png
from app.models import ClusterMetaResponse, HealthResponse
from shared.ir_config import CLUSTERING_URL, INDEX_DIR

app = FastAPI(
    title="Clustering Service",
    version="1.0",
    description="Document clustering visualization for IR Project 2026 (Task 15)",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    info = get_health_info(INDEX_DIR)
    ready = info["cluster_artifacts_ready"]
    return HealthResponse(
        status="healthy" if ready else "degraded",
        service="clustering_service",
        clustering_url=CLUSTERING_URL,
        cluster_artifacts_ready=ready,
        document_count=info.get("document_count"),
        n_clusters=info.get("n_clusters"),
        embedding_model=info.get("embedding_model"),
        missing_artifacts=info.get("missing_artifacts", []),
    )


@app.get("/cluster/meta", response_model=ClusterMetaResponse)
def cluster_meta():
    if not cluster_artifacts_ready(INDEX_DIR):
        raise HTTPException(
            status_code=503,
            detail="Cluster artifacts not ready. Run: python scripts/run_cluster_precompute.py",
        )
    meta = build_cluster_meta(INDEX_DIR)
    return ClusterMetaResponse(**meta)


@app.get("/cluster/comparison")
def cluster_comparison():
    if not cluster_artifacts_ready(INDEX_DIR):
        raise HTTPException(
            status_code=503,
            detail="Cluster artifacts not ready. Run: python scripts/run_cluster_precompute.py",
        )

    coords, labels, _ = load_tsne_data(INDEX_DIR)
    manifest = load_cluster_manifest(INDEX_DIR)
    title = (
        f"Cluster Visualization (n={manifest.get('total_docs', '?')}, "
        f"viz={manifest.get('viz_sample_size', len(labels))})"
    )
    png_bytes = render_cluster_png(coords, labels, title=title)
    return Response(content=png_bytes, media_type="image/png")
