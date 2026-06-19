import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_service.app.core.search_engine import (
    BM25SearchEngine,
    EmbeddingSearchEngine,
    HybridSearchEngine,
)
from shared.ir_config import INDEX_DIR, SERIAL_HYBRID_TOP_N

print("--- Hybrid representation smoke test ---")

bm25_eng = BM25SearchEngine(index_dir=INDEX_DIR)
emb_eng = EmbeddingSearchEngine(index_dir=INDEX_DIR)
hybrid_eng = HybridSearchEngine(bm25_engine=bm25_eng, embedding_engine=emb_eng)

tokens = ["system", "hospital"]
raw_text = "healthcare clinics and digital medicine application"

print("\n[test] Hybrid parallel (RRF):")
parallel_res = hybrid_eng.search_parallel(query_tokens=tokens, query_text=raw_text)
print(parallel_res)

print(f"\n[test] Hybrid serial (top_n={SERIAL_HYBRID_TOP_N}):")
serial_res = hybrid_eng.search_serial(
    query_tokens=tokens,
    query_text=raw_text,
    top_n_filter=SERIAL_HYBRID_TOP_N,
)
print(serial_res)

print("\n[done] Hybrid smoke test complete.")
