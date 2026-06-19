# Task 06 - Query Matching & Ranking

## Requirement Description
Task 6 requires building a subsystem to match query representations with document representations and rank results by highest similarity scores, using the appropriate matching method per representation model (e.g. VSM/embedding → cosine similarity, BM25 → BM25 formula).

## What Was Implemented
- Dedicated matching package: `retrieval_service/app/core/matching/`
- `MatcherRegistry` routes each `representation_mode` to a matcher
- `QueryRepresentation` built once after preprocessing (tokens + raw text)
- `Ranker` applies stable descending sort with tie-breaking by `doc_id`
- Matching methods per mode:

| Mode | Matching method |
|------|-----------------|
| `vsm` | Cosine similarity (log-TF-IDF dot product / norms) |
| `bm25` | BM25 term scoring (sum over query terms) |
| `embedding` | Cosine similarity (normalized embeddings) |
| `hybrid_parallel` | RRF fusion of BM25 + embedding rank lists |
| `hybrid_serial` | BM25 top-N filter → cosine rerank |

- Performance backends for embedding: `loop`, `numpy` batch, `faiss` ANN (built at index time when doc count ≥ `FAISS_THRESHOLD`)
- API metadata: `matcher`, `matching_method`, `params`, split timing (`encode_ms`, `match_ms`, `rank_ms`)
- `GET /matchers` lists supported modes and methods
- Empty-result reasons: `no_lexical_overlap`, `embedding_model_unavailable`, etc.

## Relevant Files and Components
- `retrieval_service/app/core/matching/` — matchers, registry, ranker
- `retrieval_service/app/core/search_engine.py` — index loading, BM25/embedding scoring, FAISS
- `retrieval_service/app/main.py` — thin orchestration (`preprocess → match → rank`)
- `shared/ir_config.py` — `RRF_K`, `MATCHER_METADATA`, FAISS settings
- `shared/index_builder.py` — FAISS index build on save
- `evaluation_service/` — per-mode evaluation via `POST /search`
- `tests/test_matchers.py` — unit tests with deterministic fixture
- `app_ui.py` — displays `matching_method`, params, `top_k`

## Algorithms and Techniques
- **VSM**: query vector from log-TF-IDF; score = dot(q, d) / (||q|| × ||d||)
- **BM25**: runtime scoring with configurable `k1`, `b`
- **Embedding**: `SentenceTransformer` encode with `normalize_embeddings=True`; cosine ≡ dot product when normalized
- **Hybrid parallel**: RRF with `k_rrf` (default 60)
- **Hybrid serial**: BM25 retrieves top `top_n_filter` (default 100), embedding reranks
- **FAISS**: `IndexFlatIP` on L2-normalized vectors for large collections

## Inputs and Outputs
- **Input**: raw query, `representation_mode`, optional `k1`, `b`, `top_n_filter`, `k_rrf`, `top_k`
- **Output**: ranked `{doc_id: score}` map plus matcher metadata and timing

## Design Decisions and Data Flow
1. Task 4 ends at query preprocessing; Task 6 starts at matching.
2. Matchers delegate to `search_engine` for index access; logic is explicit per mode.
3. Ranking truncation (`top_k`) is separate from matching (full scores computed first, except FAISS uses k at search).
4. Query refinement (Task 5) feeds a better query into the same matchers — no duplicate ranking logic.

## Evaluation
Run baseline evaluation for all modes:

```powershell
python -m evaluation_service.run --scale preval --max-queries 100 --output-dir evaluation_results
```

Reports include `matcher_meta` per mode (matching method and params).

## IR Quality Assessment
- **Correctness**: VSM uses full cosine; BM25 and hybrid modes tested with fixtures
- **Scalability**: NumPy batch for medium scale; FAISS for ≥10K docs
- **Observability**: Split timing and `/matchers` endpoint for report screenshots
- **Evaluation**: Wired to `evaluation_service` with outputs in `evaluation_results/`

## Done Criteria (met)
1. Every mode uses documented matching function via `MatcherRegistry`
2. Results sorted descending with deterministic ties
3. Query/doc representation consistency enforced
4. Unit tests pass for all matchers
5. Evaluation pipeline includes matcher metadata
6. UI shows matching method and parameters
