"""Orchestrate optional query refinement followed by retrieval search."""

from typing import Any, Dict, List, Optional

import requests

from shared.ir_config import RETRIEVAL_URL, refine_url


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
