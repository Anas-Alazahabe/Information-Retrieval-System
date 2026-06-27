# Task 15 - Documents Clustering

## Requirement Description

Additional feature #15: group indexed documents into semantic clusters using embedding vectors, with offline K-Means training and t-SNE visualization exposed via a dedicated SOA service.

## What Was Implemented

### Clustering service (`clustering_service`, port 8005)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Service status + whether cluster artifacts exist |
| `GET /cluster/meta` | Cluster sizes and sample document IDs per cluster |
| `GET /cluster/comparison` | PNG t-SNE scatter plot (cached coordinates) |

### Artifacts (written to `index_data/` by precompute)

| File | Purpose |
|------|---------|
| `cluster_model.pkl` | Fitted K-Means / MiniBatchKMeans model |
| `all_labels.npy` | Cluster label per document |
| `cluster_doc_ids.json` | Ordered doc IDs aligned with labels |
| `cluster_manifest.json` | Metadata (n_clusters, viz sample size, timestamp) |
| `tsne_coords.npy` | Cached 2D coordinates for visualization subsample |
| `tsne_labels.npy` | Labels for visualization subsample |
| `tsne_doc_ids.json` | Doc IDs for visualization subsample |

**Input:** `embeddings_index.json` from the indexing pipeline. For large indexes (10K+ docs with FAISS), precompute loads vectors from `embeddings.faiss` + `embeddings_id_map.json` to avoid loading the full JSON into memory.

### Precompute (`scripts/run_cluster_precompute.py`)

```powershell
python scripts/run_cluster_precompute.py
# optional: --index-dir path --max-k 10 --viz-max-points 5000
```

- Uses `MiniBatchKMeans` when document count exceeds 10,000
- Stratified subsample (default 5,000 points) for t-SNE visualization
- Run **after** index build; does not block indexing

### UI (`app_ui.py` + `ui/clustering.py`)

- Section at bottom of main page: cluster stats, t-SNE plot, per-cluster sample doc IDs
- Health status in sidebar technical details

### Port note

Clustering uses **8005**. Future RAG (Task 10 plan) also reserved 8005 — if both are needed, move RAG to 8006.

## Relevant Files

- `clustering_service/app/main.py`
- `clustering_service/app/core/precompute.py`
- `clustering_service/app/core/loader.py`
- `clustering_service/app/core/visualize.py`
- `shared/ir_config.py` (`CLUSTERING_URL`, `CLUSTER_ARTIFACT_FILES`)
- `scripts/run_cluster_precompute.py`
- `ui/clustering.py`
- `tests/test_clustering_*.py`

## Workflow

```powershell
# 1. Build index (produces embeddings_index.json)
python -m indexing_service.app.core.indexer --scale dev

# 2. Precompute clusters
python scripts/run_cluster_precompute.py

# 3. Start stack (includes clustering on 8005)
.\scripts\start_stack.ps1
```

Browser test: `http://127.0.0.1:8005/cluster/comparison`

## Limitations

- Visualization uses a subsample for large indexes (not all 200K points plotted)
- Clustering is **not** integrated into search ranking (visualization + metadata only)
- Requires `embeddings_index.json`; indexes without embeddings cannot be clustered

## IR Quality Assessment

- **SOA**: Independent FastAPI service; retrieval unchanged
- **Scalability**: MiniBatchKMeans + cached t-SNE for large corpora
- **Demonstrability**: UI plot + `/cluster/meta` JSON for reports
