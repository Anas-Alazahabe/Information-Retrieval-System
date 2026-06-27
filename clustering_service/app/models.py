from typing import Dict, List, Optional

from pydantic import BaseModel


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    sample_doc_ids: List[str]


class ClusterMetaResponse(BaseModel):
    document_count: int
    n_clusters: int
    viz_sample_size: int
    embedding_model: Optional[str] = None
    clusters: List[ClusterSummary]


class HealthResponse(BaseModel):
    status: str
    service: str
    clustering_url: str
    cluster_artifacts_ready: bool
    document_count: Optional[int] = None
    n_clusters: Optional[int] = None
    embedding_model: Optional[str] = None
    missing_artifacts: List[str] = []
