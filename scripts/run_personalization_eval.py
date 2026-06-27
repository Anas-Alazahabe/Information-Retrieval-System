"""Simulated-user personalization evaluation (baseline vs personalized re-ranking)."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

from evaluation_service.app.metrics import aggregate_metrics, evaluate_ranked_list
from shared.ir_config import (
    EVAL_DATASET_NAME,
    PERSONALIZATION_RERANK_POOL,
    PERSONALIZATION_URL,
    RETRIEVAL_URL,
)
from shared.ir_datasets_patch import patch_ir_datasets_tsv_utf8
from shared.search_pipeline import log_personalization_query_event

HEALTH_KEYWORDS = frozenset(
    {"health", "medical", "disease", "doctor", "hospital", "symptoms", "cancer", "diabetes"}
)
TECH_KEYWORDS = frozenset(
    {"computer", "software", "python", "programming", "database", "javascript", "linux", "code"}
)

SYNTHETIC_USERS = {
    "sim_health": HEALTH_KEYWORDS,
    "sim_tech": TECH_KEYWORDS,
}

METRIC_KEYS = ("map", "recall", "precision_at_10", "ndcg_at_10")


def _load_queries_and_qrels(dataset_name: str):
    patch_ir_datasets_tsv_utf8()
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels_map: Dict[str, Dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qrels_map.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return queries, qrels_map


def _bucket_query(text: str) -> Optional[str]:
    tokens = set(text.lower().split())
    if tokens & HEALTH_KEYWORDS:
        return "sim_health"
    if tokens & TECH_KEYWORDS:
        return "sim_tech"
    return None


def _bucket_queries(
    query_items: List[Tuple[str, str]],
) -> Dict[str, List[Tuple[str, str]]]:
    buckets: Dict[str, List[Tuple[str, str]]] = {name: [] for name in SYNTHETIC_USERS}
    for query_id, text in query_items:
        bucket = _bucket_query(text)
        if bucket:
            buckets[bucket].append((query_id, text))
    return buckets


def _search(
    query_text: str,
    mode: str,
    top_k: int,
    retrieval_url: str,
) -> Dict[str, float]:
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
    return payload.get("results", {})


def _personalized_search(
    user_id: str,
    query_text: str,
    mode: str,
    display_top_k: int,
    retrieval_url: str,
    personalization_url: str,
) -> Dict[str, float]:
    base_results = _search(
        query_text,
        mode,
        max(display_top_k, PERSONALIZATION_RERANK_POOL),
        retrieval_url,
    )
    if not base_results:
        return {}

    response = requests.post(
        f"{personalization_url.rstrip('/')}/personalize/rerank",
        json={
            "user_id": user_id,
            "query_text": query_text,
            "results": base_results,
        },
        timeout=60,
    )
    response.raise_for_status()
    reranked = response.json().get("results", base_results)
    return dict(list(reranked.items())[:display_top_k])


def _warmup_user(
    user_id: str,
    warmup_items: List[Tuple[str, str]],
    qrels_map: Dict[str, Dict[str, int]],
    personalization_url: str,
) -> None:
    requests.delete(
        f"{personalization_url.rstrip('/')}/profile/{user_id}",
        timeout=10,
    )
    for query_id, query_text in warmup_items:
        log_personalization_query_event(
            user_id,
            query_text,
            personalization_url=personalization_url,
        )
        relevant_docs = [
            doc_id
            for doc_id, grade in qrels_map.get(query_id, {}).items()
            if grade > 0
        ]
        if relevant_docs:
            requests.post(
                f"{personalization_url.rstrip('/')}/events/click",
                json={
                    "user_id": user_id,
                    "doc_id": relevant_docs[0],
                    "query_text": query_text,
                },
                timeout=10,
            )


def _evaluate_mode_baseline(
    mode: str,
    test_items: List[Tuple[str, str, str]],
    qrels_map: Dict[str, Dict[str, int]],
    *,
    top_k: int,
    retrieval_url: str,
) -> Dict:
    per_query = []
    for query_id, query_text, _user_id in test_items:
        qrels = qrels_map.get(query_id, {})
        if not qrels:
            continue
        results = _search(query_text, mode, top_k, retrieval_url)
        per_query.append(evaluate_ranked_list(list(results.keys()), qrels, k=top_k))
    return aggregate_metrics(per_query)


def _evaluate_mode_personalized(
    mode: str,
    test_items: List[Tuple[str, str, str]],
    qrels_map: Dict[str, Dict[str, int]],
    *,
    top_k: int,
    retrieval_url: str,
    personalization_url: str,
) -> Dict:
    per_query = []
    for query_id, query_text, user_id in test_items:
        qrels = qrels_map.get(query_id, {})
        if not qrels:
            continue
        results = _personalized_search(
            user_id,
            query_text,
            mode,
            top_k,
            retrieval_url,
            personalization_url,
        )
        per_query.append(evaluate_ranked_list(list(results.keys()), qrels, k=top_k))
    return aggregate_metrics(per_query)


def run_personalization_evaluation(
    *,
    dataset_name: str = EVAL_DATASET_NAME,
    modes: List[str],
    top_k: int = 10,
    max_queries_per_user: int = 50,
    warmup_queries: int = 5,
    retrieval_url: str = RETRIEVAL_URL,
    personalization_url: str = PERSONALIZATION_URL,
) -> Dict:
    queries, qrels_map = _load_queries_and_qrels(dataset_name)
    query_items = [(qid, text) for qid, text in queries.items() if qrels_map.get(qid)]
    buckets = _bucket_queries(query_items)

    report = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "max_queries_per_user": max_queries_per_user,
        "warmup_queries": warmup_queries,
        "synthetic_users": list(SYNTHETIC_USERS.keys()),
        "baseline": {"modes": {}},
        "personalized": {"modes": {}},
    }

    all_test_items: List[Tuple[str, str, str]] = []
    for user_id, items in buckets.items():
        if len(items) < warmup_queries + 1:
            continue
        warmup = items[:warmup_queries]
        test = items[warmup_queries : warmup_queries + max_queries_per_user]
        _warmup_user(user_id, warmup, qrels_map, personalization_url)
        for query_id, query_text in test:
            all_test_items.append((query_id, query_text, user_id))

    if not all_test_items:
        raise ValueError("No test queries available for synthetic users.")

    seen = set()
    deduped_test: List[Tuple[str, str, str]] = []
    for query_id, query_text, user_id in all_test_items:
        if query_id in seen:
            continue
        seen.add(query_id)
        deduped_test.append((query_id, query_text, user_id))

    for mode in modes:
        report["baseline"]["modes"][mode] = _evaluate_mode_baseline(
            mode,
            deduped_test,
            qrels_map,
            top_k=top_k,
            retrieval_url=retrieval_url,
        )
        report["personalized"]["modes"][mode] = _evaluate_mode_personalized(
            mode,
            deduped_test,
            qrels_map,
            top_k=top_k,
            retrieval_url=retrieval_url,
            personalization_url=personalization_url,
        )

    report["deltas_personalized_vs_baseline"] = {}
    for mode in modes:
        base = report["baseline"]["modes"].get(mode, {})
        pers = report["personalized"]["modes"].get(mode, {})
        report["deltas_personalized_vs_baseline"][mode] = {
            metric: round(pers.get(metric, 0.0) - base.get(metric, 0.0), 6)
            for metric in METRIC_KEYS
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run personalization simulated-user evaluation")
    parser.add_argument("--dataset", default=EVAL_DATASET_NAME)
    parser.add_argument("--scale", default="full", choices=["dev", "preval", "full"])
    parser.add_argument("--modes", default="bm25,embedding")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--warmup-queries", type=int, default=5)
    parser.add_argument("--output-dir", default=str(ROOT / "evaluation_results"))
    parser.add_argument("--retrieval-url", default=RETRIEVAL_URL)
    parser.add_argument("--personalization-url", default=PERSONALIZATION_URL)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(
        "Prerequisites: preprocessing (8000), retrieval (8002), personalization (8004), "
        "MySQL with documents + profile tables."
    )

    report = run_personalization_evaluation(
        dataset_name=args.dataset,
        modes=modes,
        top_k=args.top_k,
        max_queries_per_user=args.max_queries,
        warmup_queries=args.warmup_queries,
        retrieval_url=args.retrieval_url,
        personalization_url=args.personalization_url,
    )
    report["scale"] = args.scale

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / f"eval_personalization_baseline_{args.scale}_{timestamp}.json"
    personalized_path = output_dir / f"eval_personalization_simulated_{args.scale}_{timestamp}.json"
    summary_path = output_dir / f"personalization_ablation_summary_{args.scale}_{timestamp}.json"

    baseline_report = {
        "dataset_name": report["dataset_name"],
        "timestamp": report["timestamp"],
        "scale": args.scale,
        "modes": report["baseline"]["modes"],
        "top_k": report["top_k"],
    }
    personalized_report = {
        "dataset_name": report["dataset_name"],
        "timestamp": report["timestamp"],
        "scale": args.scale,
        "modes": report["personalized"]["modes"],
        "top_k": report["top_k"],
        "warmup_queries": report["warmup_queries"],
    }
    summary = {
        "scale": args.scale,
        "timestamp": report["timestamp"],
        "dataset_name": report["dataset_name"],
        "deltas_personalized_vs_baseline": report["deltas_personalized_vs_baseline"],
        "baseline": baseline_report,
        "personalized": personalized_report,
        "limitations": [
            "Simulated oracle clicks on qrels-relevant docs (upper bound).",
            "Test set is union of health/tech bucket queries after per-user warmup.",
        ],
    }

    for path, payload in (
        (baseline_path, baseline_report),
        (personalized_path, personalized_report),
        (summary_path, summary),
    ):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved {path}")

    print("\n=== Personalization delta vs baseline ===")
    for mode, deltas in summary["deltas_personalized_vs_baseline"].items():
        print(
            f"  {mode:12} MAP {deltas['map']:+.4f}  "
            f"Recall {deltas['recall']:+.4f}  "
            f"P@10 {deltas['precision_at_10']:+.4f}  "
            f"nDCG {deltas['ndcg_at_10']:+.4f}"
        )


if __name__ == "__main__":
    main()
