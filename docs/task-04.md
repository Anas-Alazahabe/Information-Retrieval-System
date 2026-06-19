# Task 04 - Query Processing

## Requirement Description
Task 4 requires processing user queries with preprocessing techniques consistent with document preprocessing, then representing/matching them in compatible retrieval spaces.

## What Was Implemented
- Query text is sent from retrieval service to preprocessing service (`/preprocess`) using same selected options as indexing path (lemmatization enabled, stopwords removed, no stemming by default).
- Query is routed by `representation_mode`:
  - `vsm`
  - `bm25`
  - `embedding`
  - `hybrid_parallel`
  - `hybrid_serial`
- BM25 parameters are accepted per request (`k1`, `b`) and exposed in UI sliders.

## Relevant Files and Components
- `retrieval_service/app/main.py` (query pipeline and mode dispatch)
- `retrieval_service/app/core/search_engine.py` (matching engines)
- `preprocessing_service/app/main.py` + `cleaner.py` (shared preprocessing)
- `app_ui.py` (query input + mode/parameter controls)

## Algorithms and Techniques
- VSM query vectorization with log-TF-IDF.
- BM25 runtime scoring over postings.
- Embedding query encoding and cosine similarity.
- Hybrid methods:
  - Parallel rank fusion (RRF).
  - Serial lexical retrieval then semantic reranking.

## Inputs and Outputs
- Input: raw user query, representation mode, optional BM25 params.
- Output: processed query tokens + ranked result map and result count.

## Design Decisions and Data Flow
1. Validate non-empty query.
2. Preprocess query through dedicated service.
3. Reload index engines from disk (freshness over performance).
4. Execute selected ranking strategy.
5. Return sorted results to UI.

## IR Quality Assessment
- **Dataset size appropriateness**:
  - Query-time algorithms are acceptable, but full-scan embedding scoring over all documents will not scale well at assignment dataset sizes.
- **Split/preparation correctness**:
  - Query/document preprocessing consistency is mostly maintained.
  - No explicit handling of official test query set and qrels for systematic evaluation loop.
- **Algorithm suitability**:
  - Method mix is suitable for baseline IR.
  - Missing query refinement techniques (Task 5 onward), as expected.
- **Metric relevance/calculation**:
  - No integrated MAP/Recall/P@10/nDCG measurement at query-processing stage.
- **Methodological flaws / leakage / overfitting**:
  - VSM implementation is not full cosine normalization (document norm omitted).
  - Engine reinitialization on each request adds overhead and may reduce throughput.
  - Serial hybrid candidate pool is too narrow by default.
- **IR best-practice compliance**:
  - Correct separation and compatibility intent.
  - Needs optimization and rigorous evaluation hooks before claiming robust query-processing quality.

## Observations and Recommendations
- Cache loaded indexes/models safely across requests; refresh via explicit reload endpoint.
- Use ANN/vector index (FAISS/HNSW) instead of full linear scan for embeddings.
- Add query analytics and error handling for malformed/unseen terms.
- Couple retrieval outputs to evaluation service once metrics pipeline is implemented.
