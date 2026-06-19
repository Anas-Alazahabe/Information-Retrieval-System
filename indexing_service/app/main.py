import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.index_builder import IndexBuilder
from shared.ir_config import INDEX_DIR

app = FastAPI(
    title="Indexing & Representation Service",
    version="2.0",
    description="Builds inverted indexes and saves representation artifacts to disk",
)

os.makedirs(INDEX_DIR, exist_ok=True)
builder = IndexBuilder()


class IndexBatchRequest(BaseModel):
    doc_ids: List[str]
    tokens_batch: List[List[str]]
    embeddings_batch: List[List[float]]


@app.post("/add-to-index")
def add_to_index(request: IndexBatchRequest):
    if not request.doc_ids:
        raise HTTPException(status_code=400, detail="Document list is empty.")

    try:
        indexed_count = builder.add_documents(
            request.doc_ids,
            request.tokens_batch,
            request.embeddings_batch,
        )
        return {
            "status": "success",
            "message": f"Added {indexed_count} documents to in-memory index.",
            "cached_docs": builder.total_docs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding to index: {e}")


@app.post("/save-index")
def save_index_to_disk():
    if builder.total_docs == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty in memory. Add documents before saving.",
        )

    try:
        manifest = builder.save(index_dir=INDEX_DIR)
        return {
            "status": "saved",
            "directory": INDEX_DIR,
            "total_terms": len(builder.raw_inverted_index),
            "total_documents": builder.total_docs,
            "manifest": manifest,
            "message": "All representation artifacts written successfully.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save index: {e}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "indexing_service",
        "cached_docs": builder.total_docs,
        "index_dir": INDEX_DIR,
    }
