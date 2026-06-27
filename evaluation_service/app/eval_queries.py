"""Shared query selection for offline evaluation on official qrels."""

import os
from typing import Dict, List, Optional, Tuple


def load_queries_and_qrels(dataset_name: str):
    """Load queries, qrels, and dataset iteration order from ir_datasets."""
    from shared.ir_datasets_patch import patch_ir_datasets_tsv_utf8

    os.environ.setdefault("PYTHONUTF8", "1")
    patch_ir_datasets_tsv_utf8()
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    query_order: List[str] = []
    queries: Dict[str, str] = {}
    for q in dataset.queries_iter():
        queries[q.query_id] = q.text
        query_order.append(q.query_id)
    qrels_map: Dict[str, Dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qrels_map.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return queries, qrels_map, query_order


def select_eval_queries(
    queries: Dict[str, str],
    qrels_map: Dict[str, Dict[str, int]],
    max_queries: Optional[int] = None,
    query_order: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Return judged queries from the first max_queries in dataset iteration order."""
    order = query_order if query_order is not None else list(queries.keys())
    if max_queries is not None:
        order = order[:max_queries]
    return [
        (query_id, queries[query_id])
        for query_id in order
        if query_id in queries and qrels_map.get(query_id)
    ]


def build_eval_protocol(
    *,
    dataset_name: str,
    num_judged_queries: int,
    max_queries: Optional[int] = None,
) -> dict:
    """Metadata documenting that evaluation uses official ir_datasets qrels."""
    return {
        "dataset": dataset_name,
        "source": "ir_datasets official qrels",
        "excludes_ui_queries": True,
        "max_queries_requested": max_queries,
        "num_judged_queries": num_judged_queries,
    }
