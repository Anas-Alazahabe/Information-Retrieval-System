# Task 02 - Document Representation

## Requirement Description
Task 2 requires representing documents using:
- VSM TF-IDF
- Embeddings
- BM25
- Hybrid representation in both:
  - Parallel mode (with fusion/scoring method)
  - Serial mode
It also requires BM25 parameter handling and hybrid selection through UI/report.

## What Was Implemented
- VSM TF-IDF weights are generated and saved in index artifacts.
- BM25 postings store `tf` and `doc_len`, enabling runtime scoring with configurable `k1` and `b`.
- Embedding vectors generated with `SentenceTransformer("all-MiniLM-L6-v2")`.
- Hybrid retrieval:
  - Parallel: Reciprocal Rank Fusion (RRF) between BM25 and embedding ranked lists.
  - Serial: BM25 pre-filter then embedding re-ranking.
- UI supports mode switching and BM25 sliders (`k1`, `b`).

## Relevant Files and Components
- `indexing_service/app/core/indexer.py` (representation generation/saving)
- `indexing_service/app/main.py` (index artifact service API)
- `retrieval_service/app/core/search_engine.py` (BM25/Embedding/Hybrid engines)
- `retrieval_service/app/main.py` (VSM runtime and mode routing)
- `app_ui.py` (user controls)
- `test_vsm.py` (basic hybrid test script)

## Algorithms, Models, and Techniques
- VSM:
  - `idf = log10(N/df)`
  - `doc_weight = (1 + log10(tf)) * idf`
  - Query uses same log-TF-IDF weighting.
- BM25:
  - Dynamic formula at query time with per-query `k1` and `b`.
- Embedding:
  - Dense vector encoding via MiniLM.
  - Cosine similarity scoring.
- Hybrid:
  - Parallel rank fusion (RRF).
  - Serial two-stage retrieval/re-ranking.

## Inputs and Outputs
- Inputs: processed tokens (lexical) and raw text (embedding), selected mode, BM25 parameters.
- Outputs: ranked `{doc_id: score}` maps.

## IR Quality Assessment
- **Dataset size appropriateness**:
  - Selected algorithms are valid for large IR datasets.
  - Current indexed data is tiny demo scale; not enough to validate effectiveness/robustness.
- **Split/preparation correctness**:
  - No evidence of train/validation/test separation for representation tuning.
  - Single-dataset setup is acceptable under the supervisor-approved exception; however, robust query/qrels-based splits and evaluation protocol are still not demonstrated.
- **Algorithm suitability**:
  - Strong baseline mix (lexical + semantic + hybrid).
  - Good choice to include both serial and parallel hybrid strategies.
- **Metric relevance/calculation**:
  - Representation outputs exist, but no evaluation metrics linked to each representation yet.
- **Methodological flaws / leakage / overfitting**:
  - No obvious leakage in representation formulas.
  - Potential mismatch: VSM cosine denominator omits document norm; this biases scores toward heavier docs/terms.
  - Serial hybrid default `top_n_filter=2` is too small and may hurt recall drastically.
- **IR best-practice compliance**:
  - Good architectural direction.
  - Needs full-scale benchmarking, proper tuning protocol, and calibrated candidate-set sizes.

## Observations and Recommendations
- Implement true cosine normalization for VSM (query norm + doc norm).
- Increase/tune serial rerank candidate pool (`top_n_filter`, e.g., 100-1000 based on dataset size).
- Add configurable fusion alternatives (weighted sum, score normalization methods) for comparison.
- Persist embedding model/config metadata with indexes for reproducibility.
