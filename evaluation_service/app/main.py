import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import locale  # أضيفي هذه

locale.getpreferredencoding = lambda *args, **kwargs: "utf-8"

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation_service.app.eval_queries import build_eval_protocol, load_queries_and_qrels, select_eval_queries
from evaluation_service.app.metrics import aggregate_metrics, evaluate_ranked_list
from shared.ir_config import EVAL_DATASET_NAME, REFINEMENT_URL, RETRIEVAL_URL, VALID_REPRESENTATION_MODES
from shared.search_pipeline import search_with_optional_refinement

app = FastAPI(title="Evaluation Service", version="1.0")

DEFAULT_REFINEMENT_TECHNIQUES = ["query_preprocess", "prf", "synonyms"]


class EvaluateRequest(BaseModel):
    """إعدادات طلب التقييم عبر API."""

    dataset_name: str = EVAL_DATASET_NAME
    representation_modes: List[str] = Field(default_factory=lambda: list(VALID_REPRESENTATION_MODES))
    top_k: int = 10
    max_queries: Optional[int] = 50
    retrieval_url: str = RETRIEVAL_URL
    use_refinement: bool = False
    refinement_techniques: List[str] = Field(default_factory=lambda: list(DEFAULT_REFINEMENT_TECHNIQUES))
    refinement_url: str = REFINEMENT_URL


def _fetch_search_payload(
    *,
    query_text: str,
    mode: str,
    top_k: int,
    retrieval_url: str,
    use_refinement: bool,
    refinement_techniques: List[str],
    max_attempts: int = 5,
) -> Dict:
    """Call retrieval (optionally via refinement) with retries on transient failures."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            if use_refinement:
                pipeline_result = search_with_optional_refinement(
                    raw_query=query_text,
                    representation_mode=mode,
                    use_refinement=True,
                    techniques=refinement_techniques,
                    previous_queries=[],
                    top_k=top_k,
                    retrieval_url=retrieval_url,
                    search_timeout=120,
                    refine_timeout=60,
                )
                return pipeline_result["search"]
            response = requests.post(
                f"{retrieval_url.rstrip('/')}/search",
                json={
                    "query": query_text,
                    "representation_mode": mode,
                    "top_k": top_k,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code not in (500, 502, 503, 504):
                raise
            last_exc = exc
        if attempt < max_attempts:
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"Retrieval unavailable after {max_attempts} attempts: {last_exc}") from last_exc


def _load_queries_and_qrels(dataset_name: str):
    return load_queries_and_qrels(dataset_name)


def run_evaluation(
    dataset_name: str = EVAL_DATASET_NAME,
    representation_modes: Optional[List[str]] = None,
    top_k: int = 10,
    max_queries: Optional[int] = 50,
    retrieval_url: str = RETRIEVAL_URL,
    use_refinement: bool = False,
    refinement_techniques: Optional[List[str]] = None,
    refinement_url: str = REFINEMENT_URL,
) -> Dict:
    """تنفيذ التقييم الكامل لكل أنماط الاسترجاع المطلوبة."""
    representation_modes = representation_modes or list(VALID_REPRESENTATION_MODES)
    refinement_techniques = refinement_techniques or list(DEFAULT_REFINEMENT_TECHNIQUES)

    queries, qrels_map, query_order = _load_queries_and_qrels(dataset_name)
    query_items = select_eval_queries(
        queries, qrels_map, max_queries, query_order=query_order
    )

    if not query_items:
        raise ValueError(f"No judged queries found for dataset {dataset_name}")

    report = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "max_queries": max_queries,
        "use_refinement": use_refinement,
        "refinement_techniques": refinement_techniques if use_refinement else [],
        "eval_protocol": build_eval_protocol(
            dataset_name=dataset_name,
            num_judged_queries=len(query_items),
            max_queries=max_queries,
        ),
        "modes": {},
    }

    for mode in representation_modes:
        if mode not in VALID_REPRESENTATION_MODES:
            continue

        per_query = []
        matcher_meta = None
        for query_id, query_text in query_items:
            qrels = qrels_map[query_id]

            try:
                payload = _fetch_search_payload(
                    query_text=query_text,
                    mode=mode,
                    top_k=top_k,
                    retrieval_url=retrieval_url,
                    use_refinement=use_refinement,
                    refinement_techniques=refinement_techniques,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Retrieval failed for query {query_id} mode {mode}: {exc}"
                ) from exc

            ranked_docs = list(payload.get("results", {}).keys())
            per_query.append(evaluate_ranked_list(ranked_docs, qrels, k=top_k))
            if matcher_meta is None:
                matcher_meta = {
                    "matcher": payload.get("matcher"),
                    "matching_method": payload.get("matching_method"),
                    "params": payload.get("params"),
                }

        mode_report = aggregate_metrics(per_query)
        if matcher_meta:
            mode_report["matcher_meta"] = matcher_meta
        report["modes"][mode] = mode_report

    return report


@app.post("/evaluate")
def evaluate_endpoint(request: EvaluateRequest):
    """Endpoint لتشغيل التقييم وإرجاع تقرير المقاييس."""
    try:
        report = run_evaluation(
            dataset_name=request.dataset_name,
            representation_modes=request.representation_modes,
            top_k=request.top_k,
            max_queries=request.max_queries,
            retrieval_url=request.retrieval_url,
            use_refinement=request.use_refinement,
            refinement_techniques=request.refinement_techniques,
            refinement_url=request.refinement_url,
        )
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health_check():
    """فحص جاهزية خدمة التقييم."""
    return {"status": "healthy", "service": "evaluation_service"}
