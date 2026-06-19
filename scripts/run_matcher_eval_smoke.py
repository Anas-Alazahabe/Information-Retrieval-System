"""Smoke evaluation using matchers directly (no HTTP). Writes to evaluation_results/."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_service.app.metrics import aggregate_metrics, evaluate_ranked_list
from retrieval_service.app.core.matching.base import MatchParams, QueryRepresentation, Ranker
from retrieval_service.app.core.matching.registry import MatcherRegistry
from retrieval_service.app.core.search_engine import BM25SearchEngine, EmbeddingSearchEngine
from shared.index_store import JsonIndexStore
from shared.ir_config import VALID_REPRESENTATION_MODES

FIXTURE = ROOT / "tests" / "fixtures" / "matcher_index"
QUERIES = [
    ("q1", "hospital system", ["hospital", "system"]),
    ("q2", "security", ["security"]),
]
EMBEDDING_MODES = frozenset({"embedding", "hybrid_parallel", "hybrid_serial"})


def main():
    store = JsonIndexStore(str(FIXTURE))
    vsm_cache = {}
    bm25 = BM25SearchEngine(store=store)
    embedding = EmbeddingSearchEngine(store=store)
    registry = MatcherRegistry(store, bm25, embedding, vsm_cache)

    report = {
        "dataset_name": "matcher-fixture-smoke",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_k": 10,
        "max_queries": len(QUERIES),
        "use_refinement": False,
        "modes": {},
    }

    qrels = {
        "q1": {"doc1": 1, "doc3": 1},
        "q2": {"doc2": 1, "doc3": 1},
    }

    mock_model = MagicMock()
    mock_model.encode.return_value = [1.0, 0.0, 0.0, 0.0]

    for mode in VALID_REPRESENTATION_MODES:
        matcher = registry.get(mode)
        per_query = []
        for qid, text, tokens in QUERIES:
            query = QueryRepresentation(raw_text=text, tokens=tokens, mode=mode)
            if mode in EMBEDDING_MODES:
                with patch.object(embedding, "model", mock_model):
                    with patch.object(embedding, "_lazy_load_model", lambda: None):
                        result = matcher.match(query, MatchParams(top_k=10))
            else:
                result = matcher.match(query, MatchParams(top_k=10))
            ranked = list(Ranker.rank(result.scores, 10).keys())
            per_query.append(evaluate_ranked_list(ranked, qrels.get(qid, {}), k=10))

        mode_report = aggregate_metrics(per_query)
        mode_report["matcher_meta"] = {
            "matcher": matcher.mode,
            "matching_method": matcher.matching_method,
        }
        report["modes"][mode] = mode_report

    out_dir = ROOT / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_matcher_smoke.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
