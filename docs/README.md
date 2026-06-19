# IR Project 2026 - Technical Overview

## Project Objectives
- Build a service-oriented Information Retrieval system in Python.
- Support multiple document/query representations (VSM TF-IDF, BM25, Embeddings, Hybrid).
- Process user queries in natural language and return ranked document IDs.
- Provide a simple UI to switch retrieval mode and tune BM25 parameters.

## Current Architecture
- `preprocessing_service`: FastAPI service for text normalization, tokenization, stopword removal, optional stemming/lemmatization.
- `indexing_service`: Builds and saves index artifacts (`vsm_index`, `bm25_index`, `embeddings_index`, `metadata`).
- `retrieval_service`: Loads index artifacts and executes query-time matching/ranking via `MatcherRegistry`.
- `evaluation_service`: Runs MAP/Recall/P@10/nDCG per representation mode.
- `app_ui.py`: Streamlit interface that calls retrieval API.

## Technologies and Libraries
- Core: Python, FastAPI, Pydantic, Requests, JSON.
- NLP: NLTK (stopwords, Porter stemmer), spaCy (lemmatization if model available).
- Semantic retrieval: `sentence-transformers` (`all-MiniLM-L6-v2`), FAISS for ANN search.
- UI: Streamlit.
- Dataset access: `ir_datasets` (implemented in indexer script path).

## Dataset Overview (As Implemented vs Assignment)
- Original assignment text states **two datasets** from `ir-datasets.com`, each with large scale (stated as >200K docs) and available test queries/qrels.
- Supervisor-approved project scope uses **one dataset** instead (accepted exception).
- Code contains a pathway for `ir_datasets.load(dataset_name)` and mentions `msmarco-passage`.
- Generated artifacts currently present in repo are very small toy/demo indexes (3-6 docs), not benchmark-scale collections.
- No full qrels-based evaluation pipeline is currently implemented in visible code.

## Execution Workflow
1. Start `preprocessing_service` (`/preprocess`, `/preprocess-batch`).
2. Run indexing logic (`DatasetIndexer`) to:
   - Load docs from dataset.
   - Generate embeddings.
   - Call preprocessing batch API.
   - Build inverted structures and metadata.
   - Save `index_data/*.json`.
3. Start `retrieval_service`:
   - Receive query and mode.
   - Preprocess query using preprocessing API.
   - Dispatch to VSM/BM25/Embedding/Hybrid engines.
4. Run Streamlit UI and send searches to retrieval service.

## Data and Control Flow
- **Index-time flow**: raw document text -> preprocessing service -> cleaned tokens -> inverted index/BM25 stats + embeddings -> JSON artifacts.
- **Query-time flow**: raw query -> preprocessing service -> query tokens -> selected ranking engine -> sorted scores by document ID.
- **Hybrid parallel**: fuse BM25 and embedding rank lists with RRF.
- **Hybrid serial**: retrieve with BM25, then re-rank a top subset via embedding similarity.

## Alignment Snapshot (Tasks 1-6)
- Task 1 (preprocessing): implemented as a separate service, generally correct for English-only baseline.
- Task 2 (representations): VSM, BM25, Embedding, and hybrid modes implemented.
- Task 3 (indexing): inverted and representation artifacts saved on disk; FAISS built at scale.
- Task 4 (query processing): query preprocessing + representation routing implemented.
- Task 5 (query refinement): optional refinement service before retrieval.
- Task 6 (matching & ranking): explicit matcher module, registry, ranker, evaluation hooks.

## Developer guides
- **`docs/developer-guide.md`** — setup, indexing, services, troubleshooting (English summary)
- **`docs/ar/developer-guide.md`** — full Arabic run guide
- **`docs/ar/implementation-overview.md`** — architecture and code walkthrough (Arabic)

Refer to:
- `docs/task-01.md`
- `docs/task-02.md`
- `docs/task-03.md`
- `docs/task-04.md`
- `docs/task-05-implementation-plan.md`
- `docs/task-06.md`
- `docs/task-06-implementation-plan.md`
- `docs/project-notes.md`
