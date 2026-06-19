"""Build a sorted prefix index of MS MARCO queries for UI autocomplete."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

from shared.ir_config import EVAL_DATASET_NAME, QUERY_SUGGESTIONS_PATH


def _normalize_query(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_queries(dataset_name: str) -> set[str]:
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    queries: set[str] = set()
    for query in dataset.queries_iter():
        normalized = _normalize_query(query.text)
        if normalized:
            queries.add(normalized)
    return queries


def build_index(sources: list[str]) -> dict:
    all_queries: set[str] = set()
    for source in sources:
        all_queries.update(load_queries(source))

    sorted_queries = sorted(all_queries)
    return {
        "source": ", ".join(sources),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "count": len(sorted_queries),
        "queries": sorted_queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MS MARCO query suggestion index.")
    parser.add_argument(
        "--output",
        default=QUERY_SUGGESTIONS_PATH,
        help="Output JSON path (default: index_data/query_suggestions.json)",
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also include msmarco-passage/train queries (slower, larger index).",
    )
    args = parser.parse_args()

    sources = [EVAL_DATASET_NAME]
    if args.include_train:
        sources.append("msmarco-passage/train")

    print(f"Loading queries from: {', '.join(sources)}")
    index_data = build_index(sources)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(index_data, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {index_data['count']} queries to {output_path}")


if __name__ == "__main__":
    main()
