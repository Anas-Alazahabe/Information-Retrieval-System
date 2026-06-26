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
**Terminal 4:** `streamlit run app_ui.py`

```powershell
# retrieval
cd retrieval_service
uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload

# query refinement (optional; needed for Task 5 synonyms / PRF)
cd query_refinement_service
uvicorn app.main:app --host 127.0.0.1 --port 8003 --reload
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

See `shared/ir_config.py`. Common overrides: `IR_INDEX_DIR`, `IR_INDEX_SCALE`, `IR_MAX_DOCS`, `IR_PREPROCESS_URL`, `IR_RETRIEVAL_URL`, `IR_REFINEMENT_URL`, `IR_QUERY_SUGGESTIONS`.

---

## Tests

```powershell
pytest tests/ -q
```

---

## Further reading

- [`docs/task-06.md`](task-06.md) — query matching & ranking
- [`docs/ar/implementation-overview.md`](ar/implementation-overview.md) — architecture and code walkthrough (Arabic)
- [`docs/ar/developer-guide.md`](ar/developer-guide.md) — full Arabic run guide with troubleshooting
