"""Evaluate vector store (FAISS) contribution via sparse vs dense vs hybrid modes."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

from evaluation_service.app.eval_queries import build_eval_protocol, load_queries_and_qrels, select_eval_queries
from evaluation_service.app.metrics import aggregate_metrics, evaluate_ranked_list
from shared.ir_config import EVAL_DATASET_NAME, RETRIEVAL_URL

VECTOR_STORE_MODES = ("bm25", "embedding", "hybrid_parallel", "hybrid_serial")
METRIC_KEYS = ("map", "recall", "precision_at_10", "ndcg_at_10")


def _load_queries_and_qrels(dataset_name: str):
    return load_queries_and_qrels(dataset_name)


def _fetch_faiss_status(retrieval_url: str) -> dict:
    response = requests.get(f"{retrieval_url.rstrip('/')}/health", timeout=10)
    response.raise_for_status()
    health = response.json()
    return {
        "faiss_loaded": health.get("ann_backend") == "faiss",
        "ann_backend": health.get("ann_backend"),
        "index_doc_count": health.get("index_doc_count"),
        "embedding_model": health.get("embedding_model"),
    }


def _search_with_timing(
    query_text: str,
    mode: str,
    top_k: int,
    retrieval_url: str,
) -> tuple[Dict[str, float], float, Optional[dict]]:
    start = time.perf_counter()
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
    elapsed_ms = (time.perf_counter() - start) * 1000
    payload = response.json()
    meta = {
        "matcher": payload.get("matcher"),
        "matching_method": payload.get("matching_method"),
    }
    return payload.get("results", {}), elapsed_ms, meta


def run_vector_store_evaluation(
    *,
    dataset_name: str = EVAL_DATASET_NAME,
    modes: Optional[List[str]] = None,
    top_k: int = 10,
    max_queries: Optional[int] = 50,
    retrieval_url: str = RETRIEVAL_URL,
) -> Dict:
    modes = list(modes or VECTOR_STORE_MODES)
    queries, qrels_map, query_order = _load_queries_and_qrels(dataset_name)
    query_items = select_eval_queries(
        queries, qrels_map, max_queries, query_order=query_order
    )

    if not query_items:
        raise ValueError(f"No judged queries for dataset {dataset_name}")

    faiss_status = _fetch_faiss_status(retrieval_url)
    report = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "max_queries": max_queries,
        "vector_store_modes": modes,
        "faiss_status": faiss_status,
        "eval_protocol": build_eval_protocol(
            dataset_name=dataset_name,
            num_judged_queries=len(query_items),
            max_queries=max_queries,
        ),
        "modes": {},
    }

    for mode in modes:
        per_query = []
        latencies: List[float] = []
        matcher_meta = None
        for query_id, query_text in query_items:
            qrels = qrels_map[query_id]
            results, elapsed_ms, meta = _search_with_timing(
                query_text, mode, top_k, retrieval_url
            )
            latencies.append(elapsed_ms)
            per_query.append(evaluate_ranked_list(list(results.keys()), qrels, k=top_k))
            if matcher_meta is None and meta.get("matcher"):
                matcher_meta = meta

        mode_report = aggregate_metrics(per_query)
        mode_report["mean_latency_ms"] = round(sum(latencies) / len(latencies), 2)
        mode_report["p95_latency_ms"] = round(
            sorted(latencies)[int(0.95 * len(latencies)) - 1], 2
        )
        if matcher_meta:
            mode_report["matcher_meta"] = matcher_meta
        report["modes"][mode] = mode_report

    bm25 = report["modes"].get("bm25", {})
    for mode in ("embedding", "hybrid_parallel", "hybrid_serial"):
        if mode not in report["modes"]:
            continue
        metrics = report["modes"][mode]
        report.setdefault("deltas_vs_bm25", {})[mode] = {
            key: round(metrics.get(key, 0.0) - bm25.get(key, 0.0), 6)
            for key in METRIC_KEYS
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector store (FAISS) evaluation")
    parser.add_argument("--dataset", default=EVAL_DATASET_NAME)
    parser.add_argument("--scale", default="full", choices=["dev", "preval", "full"])
    parser.add_argument("--modes", default=",".join(VECTOR_STORE_MODES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--output-dir", default=str(ROOT / "evaluation_results"))
    parser.add_argument("--retrieval-url", default=RETRIEVAL_URL)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("Prerequisites: preprocessing (8000) and retrieval (8002) with FAISS index.")

    report = run_vector_store_evaluation(
        dataset_name=args.dataset,
        modes=modes,
        top_k=args.top_k,
        max_queries=args.max_queries,
        retrieval_url=args.retrieval_url,
    )
    report["scale"] = args.scale

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"vector_store_ablation_{args.scale}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved {output_path}")
    print(f"FAISS loaded: {report['faiss_status']['faiss_loaded']}")


if __name__ == "__main__":
    main()
