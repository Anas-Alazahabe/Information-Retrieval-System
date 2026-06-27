import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))

from app.core.doc_terms import fetch_document_texts
from app.core.profile_builder import terms_from_click, terms_from_query
from app.core.profile_store import (
    get_documents_count,
    get_interest_terms,
    get_profile_summary,
    log_click_event,
    log_query_event,
    reset_profile,
    upsert_interest_terms,
)
from app.core.reranker import rerank_results
from app.models import (
    ClickEventRequest,
    EventResponse,
    ProfileResponse,
    QueryEventRequest,
    RerankRequest,
    RerankResponse,
)
from shared.db_config import check_db_connection
from shared.ir_config import PERSONALIZATION_ALPHA, PERSONALIZATION_URL

app = FastAPI(
    title="Personalization Service",
    version="1.0",
    description="User profiles and personalized result re-ranking for IR Project 2026",
)


@app.post("/events/query", response_model=EventResponse)
def record_query_event(request: QueryEventRequest):
    if not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    if not request.query_text.strip():
        raise HTTPException(status_code=400, detail="query_text is required.")

    try:
        term_weights = terms_from_query(request.query_text)
        log_query_event(request.user_id, request.query_text.strip())
        upsert_interest_terms(request.user_id, term_weights, "query")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    return EventResponse(
        user_id=request.user_id,
        terms_added=sorted(term_weights.keys()),
    )


@app.post("/events/click", response_model=EventResponse)
def record_click_event(request: ClickEventRequest):
    if not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    if not request.doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id is required.")

    try:
        doc_texts = fetch_document_texts([request.doc_id])
        content = doc_texts.get(request.doc_id, "")
        term_weights = terms_from_click(content) if content else {}
        log_click_event(request.user_id, request.doc_id, request.query_text)
        if term_weights:
            upsert_interest_terms(request.user_id, term_weights, "click")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    return EventResponse(
        user_id=request.user_id,
        terms_added=sorted(term_weights.keys()),
    )


@app.get("/profile/{user_id}", response_model=ProfileResponse)
def get_profile(user_id: str):
    try:
        interest_terms, query_count, click_count = get_profile_summary(user_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    return ProfileResponse(
        user_id=user_id,
        interest_terms=interest_terms,
        query_count=query_count,
        click_count=click_count,
    )


@app.post("/personalize/rerank", response_model=RerankResponse)
def personalize_rerank(request: RerankRequest):
    if not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    if not request.results:
        raise HTTPException(status_code=400, detail="results cannot be empty.")

    alpha = request.alpha if request.alpha is not None else PERSONALIZATION_ALPHA
    try:
        profile_terms = get_interest_terms(request.user_id)
        reranked, meta = rerank_results(
            request.results,
            profile_terms,
            alpha=alpha,
            query_text=request.query_text,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    return RerankResponse(
        personalization_applied=meta["personalization_applied"],
        alpha=meta["alpha"],
        profile_terms_used=meta["profile_terms_used"],
        boosted_docs=meta["boosted_docs"],
        results=reranked,
        explanation=meta["explanation"],
        missing_doc_count=meta.get("missing_doc_count", 0),
    )


@app.delete("/profile/{user_id}")
def delete_profile(user_id: str):
    try:
        reset_profile(user_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
    return {"status": "ok", "user_id": user_id, "message": "Profile reset."}


@app.get("/health")
def health_check():
    db_status = check_db_connection()
    documents_count = get_documents_count() if db_status["connected"] else None
    return {
        "status": "healthy" if db_status["connected"] else "degraded",
        "service": "personalization_service",
        "personalization_url": PERSONALIZATION_URL,
        "database_connected": db_status["connected"],
        "database_error": db_status["error"],
        "documents_count": documents_count,
    }
