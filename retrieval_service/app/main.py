import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.matching import (
    MatcherRegistry,
    QueryRepresentation,
    Ranker,
    default_match_params,
    list_matchers,
)
from app.core.search_engine import BM25SearchEngine, EmbeddingSearchEngine
from shared.index_store import JsonIndexStore
from shared.ir_config import (
    EMBEDDING_MODEL,
    INDEX_DIR,
    PREPROCESS_FLAGS,
    RRF_K,
    SERIAL_HYBRID_TOP_N,
    VALID_REPRESENTATION_MODES,
    preprocess_single_url,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retrieval & Ranking Service",
    version="2.1",
    description="Query processing, matching, and ranking for all representation modes",
)

index_store = JsonIndexStore(INDEX_DIR)
bm25_engine = BM25SearchEngine(store=index_store)
embedding_engine = EmbeddingSearchEngine(store=index_store)
_vsm_cache: Dict = {}
_index_mtime: float = 0.0

matcher_registry = MatcherRegistry(
    store=index_store,
    bm25_engine=bm25_engine,
    embedding_engine=embedding_engine,
    vsm_cache=_vsm_cache,
)


class SearchRequest(BaseModel):
    """شكل طلب البحث القادم من الواجهة أو أي عميل API."""

    query: str
    representation_mode: str
    k1: Optional[float] = 1.5
    b: Optional[float] = 0.75
    top_n_filter: Optional[int] = SERIAL_HYBRID_TOP_N
    top_k: Optional[int] = None
    k_rrf: Optional[int] = RRF_K
    preserve_wh_words: Optional[bool] = None


def _ensure_index_ready():
    """يتحقق من وجود ملفات الفهرسة قبل تنفيذ أي بحث."""
    if not index_store.index_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Index artifacts not found in {INDEX_DIR}. Run indexing first.",
        )


def _reload_indexes_if_changed():
    """يعيد تحميل الفهارس إذا تم تحديث ملفاتها على القرص."""
    global _index_mtime
    current_mtime = index_store.get_index_mtime()
    if current_mtime > _index_mtime:
        bm25_engine.reload()
        embedding_engine.reload()
        matcher_registry.clear_vsm_cache()
        _index_mtime = current_mtime
        logger.info("Reloaded index artifacts from %s", INDEX_DIR)


@app.on_event("startup")
def startup_load_indexes():
    """تحميل الفهارس عند بدء الخدمة لتحسين زمن أول استعلام."""
    global _index_mtime
    if index_store.index_ready():
        bm25_engine.reload()
        embedding_engine.reload()
        _index_mtime = index_store.get_index_mtime()
        logger.info("Loaded indexes from %s at startup", INDEX_DIR)
    else:
        logger.warning("Index not ready at startup: %s", INDEX_DIR)


@app.post("/reload-index")
def reload_index():
    """Endpoint لإعادة تحميل الفهارس يدويًا بعد إعادة البناء."""
    global _index_mtime
    _ensure_index_ready()
    bm25_engine.reload()
    embedding_engine.reload()
    matcher_registry.clear_vsm_cache()
    _index_mtime = index_store.get_index_mtime()
    return {"status": "reloaded", "index_dir": INDEX_DIR}


@app.get("/matchers")
def get_matchers():
    """List supported matchers and their matching methods."""
    return {"matchers": list_matchers()}


@app.post("/search")
def execute_search(request: SearchRequest):
    """المسار الرئيسي للاستعلام: preprocess -> match -> rank -> response."""
    t0 = time.perf_counter()
    _ensure_index_ready()
    _reload_indexes_if_changed()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text is empty.")

    mode = request.representation_mode.lower()
    if mode not in VALID_REPRESENTATION_MODES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported representation mode.",
                "valid_modes": list(VALID_REPRESENTATION_MODES),
            },
        )

    payload = {
        "text": request.query,
        **PREPROCESS_FLAGS,
        "preserve_wh_words": request.preserve_wh_words or False,
    }
    t_preprocess_start = time.perf_counter()

    try:
        response = requests.post(preprocess_single_url(), json=payload)
        response.raise_for_status()
        query_tokens = response.json()["tokens"]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reach preprocessing service: {e}",
        )

    preprocess_ms = round((time.perf_counter() - t_preprocess_start) * 1000, 2)

    query_repr = QueryRepresentation(
        raw_text=request.query,
        tokens=query_tokens,
        mode=mode,
    )
    params = default_match_params(
        k1=request.k1 or 1.5,
        b=request.b or 0.75,
        top_n_filter=request.top_n_filter,
        k_rrf=request.k_rrf,
        top_k=request.top_k,
    )

    matcher = matcher_registry.get(mode)
    match_result = matcher.match(query_repr, params)

    t_rank_start = time.perf_counter()
    results = Ranker.rank(match_result.scores, request.top_k)
    rank_ms = round((time.perf_counter() - t_rank_start) * 1000, 2)
    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    manifest = index_store.load_manifest()

    logger.info(
        "query mode=%s matcher=%s matching_method=%s preprocess_ms=%.2f match_ms=%.2f rank_ms=%.2f total_ms=%.2f results=%d",
        mode,
        matcher.mode,
        matcher.matching_method,
        preprocess_ms,
        match_result.match_ms,
        rank_ms,
        total_ms,
        len(results),
    )

    response_body = {
        "status": "success",
        "mode_used": mode,
        "matcher": matcher.mode,
        "matching_method": matcher.matching_method,
        "params": {
            "k1": params.k1,
            "b": params.b,
            "top_n_filter": params.top_n_filter,
            "k_rrf": params.k_rrf,
        },
        "query_tokens": query_tokens,
        "embedding_model": manifest.get("embedding_model", EMBEDDING_MODEL),
        "preprocess_flags": manifest.get("preprocessing", PREPROCESS_FLAGS),
        "total_results": len(results),
        "results": results,
        "timing": {
            "preprocess_ms": preprocess_ms,
            "encode_ms": match_result.encode_ms,
            "match_ms": match_result.match_ms,
            "rank_ms": rank_ms,
            "total_ms": total_ms,
        },
    }
    if match_result.empty_reason and not results:
        response_body["reason"] = match_result.empty_reason

    return response_body


@app.get("/health")
def health_check():
    """إرجاع حالة الخدمة وحالة توافر الفهارس."""
    manifest = index_store.load_manifest()
    metadata = index_store.load_metadata()
    return {
        "status": "healthy",
        "service": "retrieval_service",
        "index_dir": INDEX_DIR,
        "index_files_detected": index_store.index_ready(),
        "manifest_timestamp": manifest.get("timestamp"),
        "index_doc_count": metadata.get("total_docs", 0),
        "embedding_model": manifest.get("embedding_model", EMBEDDING_MODEL),
        "ann_backend": embedding_engine.ann_backend(),
        "matcher_version": "2.1",
    }
