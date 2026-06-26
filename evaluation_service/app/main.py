import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import locale # أضيفي هذه

locale.getpreferredencoding = lambda *args, **kwargs: "utf-8"

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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


def _load_queries_and_qrels(dataset_name: str):
    """تحميل الاستعلامات وqrels من ir_datasets."""
    os.environ.setdefault("PYTHONUTF8", "1")
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels_map: Dict[str, Dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qrels_map.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return queries, qrels_map


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

    queries, qrels_map = _load_queries_and_qrels(dataset_name)
    query_items = list(queries.items())
    if max_queries is not None:
        query_items = query_items[:max_queries]

    if not query_items:
        raise ValueError(f"No queries found for dataset {dataset_name}")

    report = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "max_queries": max_queries,
        "use_refinement": use_refinement,
        "refinement_techniques": refinement_techniques if use_refinement else [],
        "modes": {},
    }

    for mode in representation_modes:
        if mode not in VALID_REPRESENTATION_MODES:
            continue

        per_query = []
        matcher_meta = None
        for query_id, query_text in query_items:
            qrels = qrels_map.get(query_id, {})
            if not qrels:
                continue

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
                    payload = pipeline_result["search"]
                else:
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
                    payload = response.json()
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
