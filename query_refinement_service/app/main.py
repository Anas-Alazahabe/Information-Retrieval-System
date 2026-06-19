import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.refiner import QueryRefiner
from app.core.suggestions import catalog_status, load_catalog, prefix_suggestions
from app.models import RefineRequest, RefineResponse, SuggestResponse
from shared.ir_config import (
    QUERY_SUGGESTIONS_PATH,
    REFINEMENT_URL,
    SUGGEST_DEFAULT_LIMIT,
    SUGGEST_MIN_PREFIX_LEN,
    VALID_REFINEMENT_TECHNIQUES,
)

app = FastAPI(
    title="Query Refinement Service",
    version="1.0",
    description="Query formulation assistance and refinement for IR Project 2026",
)

refiner = QueryRefiner()


@app.post("/refine", response_model=RefineResponse)
def refine_query(request: RefineRequest):
    """يحسّن استعلام المستخدم وفق التقنيات المطلوبة."""
    try:
        result = refiner.refine(
            raw_query=request.raw_query,
            enabled_techniques=request.enabled_techniques,
            previous_queries=request.previous_queries,
            representation_mode=request.representation_mode,
        )
        return RefineResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/suggest", response_model=SuggestResponse)
def suggest_queries(
    q: str = Query("", description="Query prefix for autocomplete"),
    limit: int = Query(SUGGEST_DEFAULT_LIMIT, ge=1, le=20),
):
    """يعيد اقتراحات استعلام مطابقة للبادئة (UX-only)."""
    prefix = q.strip()
    if len(prefix) < SUGGEST_MIN_PREFIX_LEN:
        return SuggestResponse(query_prefix=prefix, suggestions=[])

    catalog = load_catalog(QUERY_SUGGESTIONS_PATH)
    suggestions = prefix_suggestions(catalog, prefix, limit=limit)
    return SuggestResponse(query_prefix=prefix, suggestions=suggestions)


@app.get("/health")
def health_check():
    """فحص صحة خدمة تحسين الاستعلام."""
    suggestion_status = catalog_status(QUERY_SUGGESTIONS_PATH)
    return {
        "status": "healthy",
        "service": "query_refinement_service",
        "refinement_url": REFINEMENT_URL,
        "valid_techniques": list(VALID_REFINEMENT_TECHNIQUES),
        "suggestions_index_loaded": suggestion_status["loaded"],
        "suggestions_count": suggestion_status["count"],
    }
