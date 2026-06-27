# Developer Guide — Running IR Project 2026

Step-by-step instructions for setting up, indexing, searching, and evaluating the system.

> Arabic version: [`docs/ar/developer-guide.md`](ar/developer-guide.md)

---

## Quick Start

### One-time setup

```powershell
cd C:\Users\Golden\Documents\ir_core_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional (better lemmatization):

```powershell
python -m spacy download en_core_web_sm
```

### Build index (first time or after data/config changes)

**Terminal 1 — preprocessing (required for indexing):**

```powershell
cd preprocessing_service
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — indexer CLI:**

```powershell
cd C:\Users\Golden\Documents\ir_core_project
python -m indexing_service.app.core.indexer --scale dev
```

Artifacts are written to `index_data/` at the project root.

### Daily search workflow

**Terminal 1:** preprocessing on port `8000`  
**Terminal 2:** retrieval on port `8002`  
**Terminal 3 (optional):** query refinement on port `8003` — required when using synonyms/PRF in the UI  
**Terminal 4 (optional):** personalization on port `8004` — required when using result personalization  
**Terminal 5 (optional):** clustering on port `8005` — required for cluster visualization  
**Terminal 6 (optional):** RAG on port `8006` — required for smart answers in the UI  
**Terminal 7:** `streamlit run app_ui.py`

```powershell
# retrieval
cd retrieval_service
uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload

# query refinement (optional; needed for Task 5 synonyms / PRF)
cd query_refinement_service
uvicorn app.main:app --host 127.0.0.1 --port 8003 --reload

# personalization (optional; needed for Task 16 result re-ranking)
cd personalization_service
uvicorn app.main:app --host 127.0.0.1 --port 8004 --reload

# clustering (optional; needed for Task 15 document clustering viz)
cd clustering_service
uvicorn app.main:app --host 127.0.0.1 --port 8005 --reload

# RAG (optional; needed for Task 10 smart answers — set GEMINI_API_KEY in .env)
cd rag_service
uvicorn app.main:app --host 127.0.0.1 --port 8006 --reload
```

WordNet is downloaded automatically on first synonym expansion. To pre-download:

```powershell
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Query suggestions (Task 5, UX-only)

Build the suggestion index once from MS MARCO dev queries:

```powershell
python scripts/build_query_suggestion_index.py
```

This writes `index_data/query_suggestions.json`. Suggestions help users pick realistic
queries in the UI; they are **not** used in qrels evaluation unless you explicitly
simulate partial queries.

Optional larger catalog (includes train split):

```powershell
python scripts/build_query_suggestion_index.py --include-train
```

Test the endpoint: `GET http://127.0.0.1:8003/suggest?q=how+to&limit=5`

Health check: `http://127.0.0.1:8002/health` → `index_files_detected: true`

---

## Personalization (Task 16)

### MySQL setup (one-time)

1. Install and start MySQL 8 locally.
2. Create database: `CREATE DATABASE ir_system CHARACTER SET utf8mb4;`
3. Set env vars (or use defaults in `shared/db_config.py`):

```powershell
$env:MYSQL_HOST="localhost"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_password"
$env:MYSQL_DATABASE="ir_system"
```

4. Populate documents (~200K):

```powershell
python migrate_to_db.py
```

5. Create personalization tables:

```powershell
python scripts/init_personalization_schema.py
```

6. Seed demo profiles (optional):

```powershell
python scripts/seed_demo_profiles.py
```

Health check: `http://127.0.0.1:8004/health` → `database_connected: true`

### One-command database setup

```powershell
# Schema only (after MySQL is running)
python scripts/init_personalization_schema.py

# Schema + document migration (~200K rows, slow)
python scripts/setup_personalization_db.py --migrate-docs

# Or migrate separately
python migrate_to_db.py --max-docs 200000
```

Credentials: copy `.env.example` to `.env` (gitignored) or set `MYSQL_*` env vars.

### Personalization evaluation (simulated users)

```powershell
python scripts/run_personalization_eval.py --scale full --max-queries 50 --modes bm25,embedding
```

Requires preprocessing (8000), retrieval (8002), personalization (8004), and MySQL running.

---

## Clustering (Task 15)

### Precompute (after index build)

Clustering reads `embeddings_index.json` from `index_data/` and writes cluster artifacts.

```powershell
python scripts/run_cluster_precompute.py
```

Optional flags: `--index-dir`, `--max-k 10`, `--viz-max-points 5000`

For large indexes (30K+), precompute may take several minutes. Visualization uses a stratified subsample (default 5,000 points).

### Start clustering service

Included in `scripts/start_stack.ps1` on port **8005**, or manually:

```powershell
cd clustering_service
uvicorn app.main:app --host 127.0.0.1 --port 8005 --reload
```

Health: `http://127.0.0.1:8005/health` → `cluster_artifacts_ready: true`  
Visualization: `http://127.0.0.1:8005/cluster/comparison`  
The Streamlit UI also shows the cluster plot at the bottom of the main page.

**Port note:** clustering uses **8005**; RAG uses **8006**.

---

## RAG (Task 10)

Optional natural-language answers grounded in retrieved MS MARCO passages via Gemini.

### Prerequisites

1. MySQL `documents` table populated (`python migrate_to_db.py` or `scripts/setup_personalization_db.py --migrate-docs`).
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (never commit `.env`).

### Start RAG service

Included in `scripts/start_stack.ps1` on port **8006**, or manually:

```powershell
cd rag_service
uvicorn app.main:app --host 127.0.0.1 --port 8006 --reload
```

Health: `http://127.0.0.1:8006/health` → `gemini_configured: true`, `database_connected: true`

### UI

Enable **تفعيل الإجابة الذكية (RAG)** in the sidebar. Ranked results remain visible; the generated answer appears above them.

RAG is **off** by default and does not affect `evaluation_service` batch metrics.

### Direct API test

```powershell
curl -X POST http://127.0.0.1:8006/generate `
  -H "Content-Type: application/json" `
  -d '{"query":"how to tie a tie","results":{"123":9.5,"456":8.1},"top_context_docs":3}'
```

---

## Is it the same every time?

| Scenario | Action |
|----------|--------|
| First run on machine | Setup + index build + start services |
| Normal development / demo | Start services only — **no re-index** |
| After re-indexing | Restart retrieval or `POST /reload-index` |
| After code change in a service | Restart that service |

Indexing is slow (minutes to hours). Search is fast (seconds).

---

## Scale modes

| `--scale` | Default max docs |
|-----------|------------------|
| `dev` | 5,000 |
| `preval` | 30,000 |
| `full` | 200,000 |

Check checkpoint progress without indexing:

```powershell
python -m indexing_service.app.core.indexer --status
python -m indexing_service.app.core.indexer --status --max-docs 100000
```

---

## Representation modes

`vsm`, `bm25`, `embedding`, `hybrid_parallel`, `hybrid_serial`

---

## Evaluation

Baseline (refinement off):

```powershell
python -m evaluation_service.run --scale dev --max-queries 20
```

With refinement (recommended stack):

```powershell
python -m evaluation_service.run --scale preval --max-queries 50 --use-refinement --refinement-techniques query_preprocess,prf,synonyms
```

Full before/after ablation (baseline + single-technique ablations + combined):

```powershell
python scripts/run_refinement_ablation.py --scale preval --max-queries 50
```

Requires preprocessing (8000), retrieval (8002), and refinement (8003) running.
History and query suggestions are **not** used in batch evaluation.

Reports saved under `evaluation_results/` (default) including matcher metadata per mode.

List matching methods: `GET http://127.0.0.1:8002/matchers`

Matcher unit tests: `pytest tests/test_matchers.py -q`

---

## Environment variables

See `shared/ir_config.py`. Common overrides: `IR_INDEX_DIR`, `IR_INDEX_SCALE`, `IR_MAX_DOCS`, `IR_PREPROCESS_URL`, `IR_RETRIEVAL_URL`, `IR_REFINEMENT_URL`, `IR_PERSONALIZATION_URL`, `IR_CLUSTERING_URL`, `IR_RAG_URL`, `IR_RAG_MODEL`, `GEMINI_API_KEY`, `IR_CLUSTER_MAX_K`, `IR_CLUSTER_VIZ_MAX`, `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`, `IR_QUERY_SUGGESTIONS`.

---

## Tests

```powershell
pytest tests/ -q
```

---

## Further reading

- [`docs/task-05.md`](task-05.md) — query refinement
- [`docs/task-10-rag-implementation-plan.md`](task-10-rag-implementation-plan.md) — RAG (Gemini)
- [`docs/task-15-clustering.md`](task-15-clustering.md) — document clustering
- [`docs/task-16-personalization.md`](task-16-personalization.md) — personalization
- [`docs/task-06.md`](task-06.md) — query matching & ranking
- [`docs/ar/implementation-overview.md`](ar/implementation-overview.md) — architecture and code walkthrough (Arabic)
- [`docs/ar/developer-guide.md`](ar/developer-guide.md) — full Arabic run guide with troubleshooting
