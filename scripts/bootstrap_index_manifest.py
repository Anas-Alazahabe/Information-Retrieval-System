"""Bootstrap manifest and doc_norms for an existing on-disk index."""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.ir_config import (
    ARTIFACT_FILES,
    DATASET_NAME,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INDEX_SCALE_MODE,
    PREPROCESS_FLAGS,
    get_git_commit,
    get_max_docs_for_scale,
    preprocess_single_url,
)

import requests


def compute_doc_norms(vsm_index):
    doc_weight_squares = {}
    for postings in vsm_index.values():
        for doc_id, weight in postings.items():
            doc_weight_squares[doc_id] = doc_weight_squares.get(doc_id, 0.0) + weight ** 2
    return {
        doc_id: round(math.sqrt(square_sum), 6)
        for doc_id, square_sum in doc_weight_squares.items()
        if square_sum > 0
    }


def fetch_preprocessing_health():
    try:
        response = requests.get(f"{preprocess_single_url().rsplit('/', 1)[0]}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def main():
    index_dir = Path(INDEX_DIR)
    metadata_path = index_dir / "metadata.json"
    vsm_path = index_dir / "vsm_index.json"
    embeddings_path = index_dir / "embeddings_index.json"

    if not metadata_path.exists() or not vsm_path.exists():
        print(f"Index not found in {index_dir}")
        return 1

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(vsm_path, "r", encoding="utf-8") as f:
        vsm_index = json.load(f)

    if "doc_norms" not in metadata:
        metadata["doc_norms"] = compute_doc_norms(vsm_index)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print("Added doc_norms to metadata.json")

    doc_lengths = metadata.get("doc_lengths", {})
    embeddings = {}
    if embeddings_path.exists():
        with open(embeddings_path, "r", encoding="utf-8") as f:
            embeddings = json.load(f)

    missing_embeddings = [
        doc_id for doc_id in doc_lengths if doc_id not in embeddings or not embeddings[doc_id]
    ]
    health = fetch_preprocessing_health()

    manifest = {
        "dataset_name": DATASET_NAME,
        "document_count": metadata.get("total_docs", 0),
        "preprocessing": {
            **PREPROCESS_FLAGS,
            "spacy_available": health.get("spacy_available", False),
            "lemmatization_mode": health.get("lemmatization_mode", "unknown"),
        },
        "embedding_model": EMBEDDING_MODEL,
        "index_scale_mode": INDEX_SCALE_MODE,
        "max_docs_cap": get_max_docs_for_scale(INDEX_SCALE_MODE),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "artifact_files": list(ARTIFACT_FILES),
        "sanity_checks": {
            "indexed_docs_count": metadata.get("total_docs", 0),
            "empty_token_docs_count": 0,
            "missing_embeddings_count": len(missing_embeddings),
            "unique_terms_count": len(vsm_index),
        },
    }

    manifest_path = index_dir / "index_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {manifest_path} ({manifest['timestamp']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
