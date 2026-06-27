"""RAG evaluation on official qrels queries (latency, citations, faithfulness proxy)."""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

_env_path = ROOT / ".env"
if _env_path.is_file():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from evaluation_service.app.eval_queries import build_eval_protocol, load_queries_and_qrels, select_eval_queries
from shared.ir_config import EVAL_DATASET_NAME, RETRIEVAL_URL
from shared.search_pipeline import search_with_rag

DEFAULT_TECHNIQUES = ["query_preprocess", "prf", "synonyms"]
DEFAULT_MODE = "hybrid_parallel"


def _load_queries_and_qrels(dataset_name: str):
    return load_queries_and_qrels(dataset_name)


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _faithfulness_overlap(answer: str, citations: List[dict]) -> float:
    """Token overlap between answer sentences and cited passage snippets."""
    if not answer or not citations:
        return 0.0
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    snippet_tokens: Set[str] = set()
    for cite in citations:
        snippet_tokens |= _tokenize(cite.get("snippet", ""))
    if not snippet_tokens:
        return 0.0
    return round(len(answer_tokens & snippet_tokens) / len(answer_tokens), 4)


def run_rag_evaluation(
    *,
    dataset_name: str = EVAL_DATASET_NAME,
    max_queries: int = 20,
    top_k: int = 10,
    representation_mode: str = DEFAULT_MODE,
    use_refinement: bool = True,
    retrieval_url: str = RETRIEVAL_URL,
) -> Dict:
    queries, qrels_map, query_order = _load_queries_and_qrels(dataset_name)
    query_items = select_eval_queries(
        queries, qrels_map, max_queries, query_order=query_order
    )

    if not query_items:
        raise ValueError(f"No judged queries for dataset {dataset_name}")

    per_query: List[dict] = []
    ranking_unchanged_count = 0

    for query_id, query_text in query_items:
        try:
            without_rag = search_with_rag(
                raw_query=query_text,
                representation_mode=representation_mode,
                use_refinement=use_refinement,
                use_personalization=False,
                use_rag=False,
                techniques=DEFAULT_TECHNIQUES,
                top_k=top_k,
                retrieval_url=retrieval_url,
            )
            with_rag = search_with_rag(
                raw_query=query_text,
                representation_mode=representation_mode,
                use_refinement=use_refinement,
                use_personalization=False,
                use_rag=True,
                techniques=DEFAULT_TECHNIQUES,
                top_k=top_k,
                retrieval_url=retrieval_url,
                rag_timeout=120,
            )
        except Exception as exc:
            per_query.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "error": str(exc),
                    "has_answer": False,
                    "num_citations": 0,
                    "latency_ms": 0.0,
                }
            )
            continue

        base_results = without_rag["search"].get("results", {})
        rag_results = with_rag["search"].get("results", {})
        if list(base_results.keys()) == list(rag_results.keys()):
            ranking_unchanged_count += 1

        rag_meta = with_rag.get("rag") or {}
        citations = rag_meta.get("citations", [])
        cited_ids = [c.get("doc_id") for c in citations if c.get("doc_id")]
        context_ids = rag_meta.get("context_doc_ids", [])
        retrieved_ids = set(base_results.keys())

        timing = rag_meta.get("timing") or {}
        total_ms = timing.get("total_ms", 0.0)

        per_query.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "has_answer": bool(rag_meta.get("answer")),
                "num_citations": len(citations),
                "cited_doc_ids": cited_ids,
                "cited_in_retrieved": all(cid in retrieved_ids for cid in cited_ids),
                "cited_in_context": all(cid in context_ids for cid in cited_ids),
                "faithfulness_overlap": _faithfulness_overlap(
                    rag_meta.get("answer", ""), citations
                ),
                "latency_ms": total_ms,
                "answer_preview": (rag_meta.get("answer") or "")[:200],
            }
        )

    n = max(len(per_query), 1)
    successful = [q for q in per_query if "error" not in q]
    latencies = sorted(q.get("latency_ms", 0) for q in successful if q.get("latency_ms", 0) > 0)
    p95_idx = max(0, int(0.95 * len(latencies)) - 1) if latencies else 0

    aggregate = {
        "num_queries": len(per_query),
        "num_successful": len(successful),
        "num_errors": len(per_query) - len(successful),
        "citation_rate": round(
            sum(1 for q in successful if q.get("num_citations", 0) > 0) / max(len(successful), 1),
            4,
        ),
        "cited_in_retrieved_rate": round(
            sum(1 for q in successful if q.get("cited_in_retrieved")) / max(len(successful), 1),
            4,
        ),
        "cited_in_context_rate": round(
            sum(1 for q in successful if q.get("cited_in_context")) / max(len(successful), 1),
            4,
        ),
        "mean_faithfulness_overlap": round(
            sum(q.get("faithfulness_overlap", 0) for q in successful) / max(len(successful), 1),
            4,
        ),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(latencies[p95_idx], 2) if latencies else 0.0,
        "ranking_unchanged_rate": round(
            ranking_unchanged_count / max(len(successful), 1), 4
        ),
    }

    return {
        "dataset_name": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "representation_mode": representation_mode,
        "use_refinement": use_refinement,
        "refinement_techniques": DEFAULT_TECHNIQUES if use_refinement else [],
        "top_k": top_k,
        "max_queries": max_queries,
        "eval_protocol": build_eval_protocol(
            dataset_name=dataset_name,
            num_judged_queries=len(query_items),
            max_queries=max_queries,
        ),
        "aggregate": aggregate,
        "per_query": per_query,
        "limitations": [
            "RAG does not change retrieval rankings; MAP/nDCG are not applicable.",
            "Faithfulness is approximated by token overlap with cited snippets.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation on official queries")
    parser.add_argument("--dataset", default=EVAL_DATASET_NAME)
    parser.add_argument("--scale", default="full", choices=["dev", "preval", "full"])
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "evaluation_results"))
    parser.add_argument("--retrieval-url", default=RETRIEVAL_URL)
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set; RAG service may fail.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(
        "Prerequisites: full stack including RAG (8006), MySQL, GEMINI_API_KEY."
    )

    report = run_rag_evaluation(
        dataset_name=args.dataset,
        max_queries=args.max_queries,
        top_k=args.top_k,
        representation_mode=args.mode,
        use_refinement=not args.no_refinement,
        retrieval_url=args.retrieval_url,
    )
    report["scale"] = args.scale

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rag_eval_{args.scale}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved {output_path}")
    print(f"Aggregate: {json.dumps(report['aggregate'], indent=2)}")


if __name__ == "__main__":
    main()
