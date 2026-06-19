import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "index_data"
sys.path.insert(0, str(ROOT))

from retrieval_service.app.core.matching.registry import MatcherRegistry
from retrieval_service.app.core.search_engine import BM25SearchEngine, EmbeddingSearchEngine
from shared.index_store import JsonIndexStore


def _compute_doc_norms(vsm_index):
    doc_weight_squares = {}
    for postings in vsm_index.values():
        for doc_id, weight in postings.items():
            doc_weight_squares[doc_id] = doc_weight_squares.get(doc_id, 0.0) + weight ** 2
    return {
        doc_id: round(math.sqrt(square_sum), 6)
        for doc_id, square_sum in doc_weight_squares.items()
        if square_sum > 0
    }


@pytest.fixture(scope="module", autouse=True)
def ensure_fixture_index():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / "index_data"
    if not (source_dir / "metadata.json").exists():
        pytest.skip("Root index_data not available for fixture bootstrap")

    for name in ("vsm_index.json", "bm25_index.json", "embeddings_index.json", "metadata.json"):
        src = source_dir / name
        dst = FIXTURE_DIR / name
        if src.exists():
            if name == "metadata.json":
                with open(src, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if "doc_norms" not in metadata:
                    with open(source_dir / "vsm_index.json", "r", encoding="utf-8") as vf:
                        vsm_index = json.load(vf)
                    metadata["doc_norms"] = _compute_doc_norms(vsm_index)
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(metadata, f)
            else:
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


@pytest.fixture
def registry():
    store = JsonIndexStore(str(FIXTURE_DIR))
    vsm_cache = {}
    bm25 = BM25SearchEngine(store=store)
    embedding = EmbeddingSearchEngine(store=store)
    return MatcherRegistry(store, bm25, embedding, vsm_cache)


def test_bm25_matcher_ranks_hospital_doc(registry):
    from retrieval_service.app.core.matching.base import MatchParams, QueryRepresentation

    matcher = registry.get("bm25")
    query = QueryRepresentation(
        raw_text="hospital system",
        tokens=["hospital", "system"],
        mode="bm25",
    )
    result = matcher.match(query, MatchParams())
    assert len(result.scores) >= 1
    top_score = next(iter(result.scores.values()))
    assert top_score > 0


def test_hybrid_serial_uses_configurable_pool(registry):
    from retrieval_service.app.core.matching.base import MatchParams, QueryRepresentation

    matcher = registry.get("hybrid_serial")
    query = QueryRepresentation(
        raw_text="hospital security system",
        tokens=["hospital"],
        mode="hybrid_serial",
    )
    result = matcher.match(query, MatchParams(top_n_filter=2))
    assert len(result.scores) <= 2


def test_hybrid_parallel_returns_ordered_results(registry):
    from retrieval_service.app.core.matching.base import MatchParams, QueryRepresentation

    matcher = registry.get("hybrid_parallel")
    query = QueryRepresentation(
        raw_text="healthcare hospital system",
        tokens=["system", "hospital"],
        mode="hybrid_parallel",
    )
    result = matcher.match(query, MatchParams())
    assert len(result.scores) >= 1
    scores = list(result.scores.values())
    assert scores == sorted(scores, reverse=True)
