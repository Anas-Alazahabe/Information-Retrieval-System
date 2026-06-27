"""Orchestrate optional query refinement, retrieval, and personalization."""

from typing import Any, Dict, List, Optional

import requests

from shared.ir_config import (
    PERSONALIZATION_ALPHA,
    PERSONALIZATION_RERANK_POOL,
    PERSONALIZATION_URL,
    RETRIEVAL_URL,
    personalize_click_event_url,
    personalize_query_event_url,
    personalize_rerank_url,
    refine_url,
)


def search_with_optional_refinement(
    raw_query: str,
    representation_mode: str,
    use_refinement: bool,
    techniques: List[str],
    *,
    previous_queries: Optional[List[str]] = None,
    k1: float = 1.5,
    b: float = 0.75,
    top_n_filter: int = 100,
    top_k: Optional[int] = None,
    refine_timeout: int = 30,
    search_timeout: int = 120,
    retrieval_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Call /refine when enabled, then POST /search with the final query."""
    refinement_meta = None
    search_query = raw_query
    preserve_wh_words = False

    if use_refinement:
        refine_resp = requests.post(
            refine_url(),
            json={
                "raw_query": raw_query,
                "enabled_techniques": techniques,
                "previous_queries": previous_queries or [],
                "representation_mode": representation_mode,
            },
            timeout=refine_timeout,
        )
        refine_resp.raise_for_status()
        refinement_meta = refine_resp.json()
        search_query = refinement_meta.get("refined_query", raw_query)
        preserve_wh_words = refinement_meta.get("preprocess_hints", {}).get(
            "preserve_wh_words", False
        )

    payload: Dict[str, Any] = {
        "query": search_query,
        "representation_mode": representation_mode,
        "k1": k1,
        "b": b,
        "top_n_filter": top_n_filter,
        "preserve_wh_words": preserve_wh_words if use_refinement else False,
    }
    if top_k is not None:
        payload["top_k"] = top_k

    search_base = (retrieval_url or RETRIEVAL_URL).rstrip("/")
    search_resp = requests.post(
        f"{search_base}/search",
        json=payload,
        timeout=search_timeout,
    )
    search_resp.raise_for_status()
    return {"search": search_resp.json(), "refinement": refinement_meta}


def log_personalization_query_event(
    user_id: str,
    query_text: str,
    *,
    personalization_url: Optional[str] = None,
    timeout: int = 10,
) -> None:
    """Log a query event to the personalization service (best-effort)."""
    base = (personalization_url or PERSONALIZATION_URL).rstrip("/")
    try:
        requests.post(
            f"{base}/events/query",
            json={"user_id": user_id, "query_text": query_text},
            timeout=timeout,
        ).raise_for_status()
    except Exception:
        pass


def log_personalization_click_event(
    user_id: str,
    doc_id: str,
    query_text: str = "",
    *,
    personalization_url: Optional[str] = None,
    timeout: int = 10,
) -> bool:
    """Log a click event; returns True on success."""
    base = (personalization_url or PERSONALIZATION_URL).rstrip("/")
    try:
        response = requests.post(
            f"{base}/events/click",
            json={
                "user_id": user_id,
                "doc_id": doc_id,
                "query_text": query_text or None,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return True
    except Exception:
        return False


def search_with_personalization(
    raw_query: str,
    representation_mode: str,
    use_refinement: bool,
    use_personalization: bool,
    techniques: List[str],
    *,
    user_id: Optional[str] = None,
    previous_queries: Optional[List[str]] = None,
    k1: float = 1.5,
    b: float = 0.75,
    top_n_filter: int = 100,
    top_k: Optional[int] = None,
    alpha: Optional[float] = None,
    refine_timeout: int = 30,
    search_timeout: int = 120,
    personalization_timeout: int = 30,
    retrieval_url: Optional[str] = None,
    personalization_url: Optional[str] = None,
    log_query_event: bool = True,
) -> Dict[str, Any]:
    """Refine (optional) -> search -> personalize rerank (optional) -> log query event."""
    search_pool = top_k
    if use_personalization and user_id:
        search_pool = max(top_k or 0, PERSONALIZATION_RERANK_POOL)

    pipeline = search_with_optional_refinement(
        raw_query=raw_query,
        representation_mode=representation_mode,
        use_refinement=use_refinement,
        techniques=techniques,
        previous_queries=previous_queries,
        k1=k1,
        b=b,
        top_n_filter=top_n_filter,
        top_k=search_pool,
        refine_timeout=refine_timeout,
        search_timeout=search_timeout,
        retrieval_url=retrieval_url,
    )

    search_payload = pipeline["search"]
    personalization_meta = None

    if (
        use_personalization
        and user_id
        and search_payload.get("status") == "success"
        and search_payload.get("results")
    ):
        rerank_resp = requests.post(
            personalize_rerank_url(personalization_url),
            json={
                "user_id": user_id,
                "query_text": raw_query,
                "results": search_payload["results"],
                "alpha": alpha if alpha is not None else PERSONALIZATION_ALPHA,
            },
            timeout=personalization_timeout,
        )
        rerank_resp.raise_for_status()
        personalization_meta = rerank_resp.json()

        reranked = personalization_meta.get("results", {})
        if top_k is not None and top_k > 0:
            reranked = dict(list(reranked.items())[:top_k])
        search_payload = {**search_payload, "results": reranked}
        search_payload["total_results"] = len(reranked)

    if use_personalization and user_id and log_query_event:
        log_personalization_query_event(
            user_id,
            raw_query,
            personalization_url=personalization_url,
        )

    return {
        "search": search_payload,
        "refinement": pipeline["refinement"],
        "personalization": personalization_meta,
    }
