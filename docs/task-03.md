# Task 03 - Indexing

## Requirement Description
Task 3 requires building one or more effective index structures (e.g., inverted index) for fast retrieval and efficient term access.

## What Was Implemented
- Inverted index structures are built from preprocessed tokens.
- Multiple persisted artifacts are created:
  - `vsm_index.json`
  - `bm25_index.json`
  - `embeddings_index.json`
  - `metadata.json`
- BM25 artifact stores posting-level `tf` and `doc_len` to allow runtime scoring with dynamic parameters.
- Metadata stores corpus stats (`total_docs`, `avg_doc_len`, `doc_lengths`, `idf_weights`).

## Relevant Files and Components
- `indexing_service/app/core/indexer.py`
- `indexing_service/app/main.py`
- Index output directories:
  - `index_data/`
  - `indexing_service/index_data/` (duplicate output location also exists)

## Algorithms and Data Structures
- Inverted index: term -> doc -> term frequency.
- TF-IDF computation for VSM artifact.
- BM25-ready posting storage.
- Dense vector store serialized in JSON.

## Inputs and Outputs
- Input: dataset docs from `ir_datasets`, preprocessing API output, embedding model vectors.
- Output: JSON index files used by retrieval service.

## Design and Data Flow
1. Iterate dataset docs (with a `max_docs` cap in current script).
2. Encode raw text to dense vectors.
3. Batch-call preprocessing service.
4. Build lexical postings and document stats.
5. Compute derived weights/statistics and serialize artifacts.

## IR Quality Assessment
- **Dataset size appropriateness**:
  - Architecture is conceptually scalable, but JSON full-load indexes are not suitable for very large corpora (>200K docs each).
- **Split/preparation correctness**:
  - Indexing handles documents only; qrels/query evaluation dataset handling is absent.
- **Algorithm suitability**:
  - Inverted index + BM25 stats is appropriate.
  - JSON as primary storage is a prototype choice, not production IR indexing.
- **Metric relevance/calculation**:
  - No indexing-specific quality checks (coverage, OOV rates, posting stats distributions).
- **Methodological flaws / leakage / overfitting**:
  - Major gap: script defaults to `max_docs=2000`, conflicting with assignment scale expectations.
  - Duplicate index directories can create inconsistency about which artifact set retrieval uses.
  - No versioned index manifest (dataset ID, preprocessing config, model hash, timestamp).
- **IR best-practice compliance**:
  - Correct baseline structure, but persistence layer and reproducibility controls are insufficient for the assignment scale.

## Observations and Recommendations
- Replace JSON indexes with compressed or database-backed storage for large datasets.
- Remove/clarify duplicate `index_data` locations and enforce one authoritative path.
- Add index manifest and validation checks (doc count match, missing embeddings, empty token rate).
- Ensure full dataset indexing (or clearly documented staged subsets for development only).
