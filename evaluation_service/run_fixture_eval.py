"""Offline evaluation against the local fixture index (no HTTP / no msmarco download).

WARNING: Demo only — uses 3 hand-made queries. Do NOT use for grading;
official metrics come from msmarco-passage/dev via evaluation_service/app/main.py.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from evaluation_service.app.metrics import aggregate_metrics, evaluate_ranked_list
from retrieval_service.app.core.search_engine import (
    BM25SearchEngine,
    EmbeddingSearchEngine,
    HybridSearchEngine,
)
from retrieval_service.app.main import search_vsm
from shared.ir_config import INDEX_DIR, VALID_REPRESENTATION_MODES

FIXTURE_QUERIES = {
    "q1": "hospital system",
    "q2": "information retrieval",
    "q3": "smart security",
}

FIXTURE_QRELS = {
    "q1": {"doc_2": 1, "doc_1": 0, "doc_3": 0},
    "q2": {"doc_1": 2, "doc_3": 1, "doc_2": 0},
    "q3": {"doc_2": 1, "doc_3": 1, "doc_1": 0},
}


def _rank(mode: str, query_text: str, tokens: list, engines) -> list:
    """تنفيذ الاسترجاع حسب النمط وإرجاع قائمة doc_ids مرتبة."""
    bm25, embedding, hybrid = engines
    if mode == "vsm":
        results = search_vsm(tokens)
    elif mode == "bm25":
        results = bm25.search(tokens)
    elif mode == "embedding":
        results = embedding.search(query_text)
    elif mode == "hybrid_parallel":
        results = hybrid.search_parallel(tokens, query_text)
    elif mode == "hybrid_serial":
        results = hybrid.search_serial(tokens, query_text)
    else:
        results = {}
    return list(results.keys())


def run_fixture_evaluation(top_k: int = 10) -> dict:
    """تشغيل تقييم سريع محلي على بيانات fixture صغيرة."""
    bm25 = BM25SearchEngine(index_dir=INDEX_DIR)
    embedding = EmbeddingSearchEngine(index_dir=INDEX_DIR)
    hybrid = HybridSearchEngine(bm25, embedding)
    engines = (bm25, embedding, hybrid)

    token_map = {
        "q1": ["hospital", "system"],
        "q2": ["information", "retrieval"],
        "q3": ["smart", "security"],
    }

    report = {
        "dataset_name": "local_fixture",
        "index_dir": INDEX_DIR,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "modes": {},
    }

    for mode in VALID_REPRESENTATION_MODES:
        per_query = []
        for qid, query_text in FIXTURE_QUERIES.items():
            ranked = _rank(mode, query_text, token_map[qid], engines)
            per_query.append(evaluate_ranked_list(ranked, FIXTURE_QRELS[qid], k=top_k))
        report["modes"][mode] = aggregate_metrics(per_query)

    return report


def main():
    """نقطة تشغيل السكربت المحلي وحفظ تقرير JSON."""
    report = run_fixture_evaluation()
    output_dir = _ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"eval_fixture_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
