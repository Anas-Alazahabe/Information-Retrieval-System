import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))

from app.core.context_builder import build_context
from app.core.gemini_client import GeminiError, generate_answer
from app.models import Citation, GenerateRequest, GenerateResponse, GenerateTiming
from shared.db_config import check_db_connection
from shared.ir_config import (
    GEMINI_API_KEY,
    RAG_DEFAULT_MODEL,
    RAG_MAX_CONTEXT_CHARS,
    RAG_TOP_CONTEXT_DOCS,
    RAG_URL,
)

app = FastAPI(
    title="RAG Service",
    version="1.0",
    description="Retrieval-Augmented Generation for IR Project 2026",
)


@app.post("/generate", response_model=GenerateResponse)
def generate_rag_answer(request: GenerateRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required.")
    if not request.results:
        raise HTTPException(status_code=400, detail="results cannot be empty.")

    top_n = request.top_context_docs or RAG_TOP_CONTEXT_DOCS
    max_chars = request.max_context_chars or RAG_MAX_CONTEXT_CHARS
    model = request.model or RAG_DEFAULT_MODEL

    t0 = time.perf_counter()
    context_data = build_context(
        request.results,
        top_context_docs=top_n,
        max_context_chars=max_chars,
    )
    fetch_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not context_data["context_doc_ids"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No passage text found for retrieved documents.",
                "missing_doc_ids": context_data["missing_doc_ids"],
                "hint": "Run migrate_to_db.py to populate the documents table.",
            },
        )

    if not GEMINI_API_KEY.strip():
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    t_gen = time.perf_counter()
    try:
        answer = generate_answer(
            request.query.strip(),
            context_data["context_text"],
            api_key=GEMINI_API_KEY,
            model=model,
        )
    except GeminiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    generate_ms = round((time.perf_counter() - t_gen) * 1000, 2)
    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    citations = []
    if request.include_citations:
        citations = [Citation(**item) for item in context_data["citations"]]

    return GenerateResponse(
        query=request.query.strip(),
        answer=answer,
        citations=citations,
        context_doc_ids=context_data["context_doc_ids"],
        missing_doc_ids=context_data["missing_doc_ids"],
        model=model,
        timing=GenerateTiming(
            fetch_ms=fetch_ms,
            generate_ms=generate_ms,
            total_ms=total_ms,
        ),
    )


@app.get("/health")
def health_check():
    db_status = check_db_connection()
    gemini_configured = bool(GEMINI_API_KEY.strip())
    healthy = gemini_configured and db_status["connected"]
    return {
        "status": "healthy" if healthy else "degraded",
        "service": "rag_service",
        "rag_url": RAG_URL,
        "gemini_configured": gemini_configured,
        "database_connected": db_status["connected"],
        "database_error": db_status.get("error"),
        "default_model": RAG_DEFAULT_MODEL,
    }
