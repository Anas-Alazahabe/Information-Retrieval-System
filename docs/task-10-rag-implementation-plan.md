# Task 10 — RAG (Retrieval-Augmented Generation) Implementation Plan

## 1. Requirement context

### Assignment (IR Project 2026 §10)

The project lists **RAG** as an optional additional feature. The core system (Tasks 1–8) must return **ranked document IDs** with standard IR metrics (MAP, Recall, P@10, nDCG). RAG extends that baseline by:

1. **Retrieving** relevant passages from the corpus (already implemented).
2. **Augmenting** an LLM prompt with those passages as context.
3. **Generating** a natural-language answer grounded in retrieved evidence.

This satisfies the assignment goal of handling natural-language queries and returning results “in natural language” — but as a **second output layer** on top of retrieval, not a replacement for ranked lists or qrels evaluation.

### Relationship to existing work

| Area | Status | RAG reuse |
|------|--------|-----------|
| Preprocessing (Task 1) | Done | Query text passed unchanged to LLM |
| Representations + retrieval (Tasks 2–4, 6) | Done | **Retrieval step** — primary RAG input |
| Query refinement (Task 5) | Done | Optional pre-step; PRF/synonyms improve recall |
| Personalization (Task 16) | Done | Optional rerank before context selection |
| Document text store | MySQL `documents` (~200K MS MARCO) | **Context source** for passages |
| LLM query rewriting | Explicitly deferred from Task 5 | Optional **query rewrite** sub-feature inside RAG service |
| Gemini API key | Available (not in repo) | Generation backend |

Task 5 plan note:

> *Neural query rewriting (LLM) — Hard to evaluate fairly; overlaps with optional §10 RAG.*

We implement LLM usage here, not in refinement, to keep qrels evaluation clean.

---

## 2. What we will implement (scope)

### In scope (defensible §10 deliverable)

1. **`rag_service`** — new FastAPI microservice (port **8005**).
2. **Context assembly** — fetch top-*N* passage texts for retrieved `doc_id`s from MySQL.
3. **Gemini generation** — grounded answer + inline citations (`[doc_id]`).
4. **Pipeline orchestration** — extend `shared/search_pipeline.py` with optional RAG step after search (and after personalization if enabled).
5. **UI** — Streamlit toggle “إجابة ذكية (RAG)” showing generated answer + source passages; keep ranked list visible.
6. **Config** — `GEMINI_API_KEY`, model name, context limits via `shared/ir_config.py`.
7. **Tests** — unit tests with mocked Gemini; integration smoke test.
8. **Documentation** — developer guide section + Arabic report subsection.

### Out of scope (first iteration)

- Replacing BM25/embedding index with a separate vector DB (§11 overlap; we already use FAISS).
- Training/fine-tuning models.
- Conversational multi-turn agents (§18).
- Automatic qrels-based MAP/nDCG for generated answers (different evaluation paradigm).
- Storing embeddings in Gemini File API / external vector store.

### Optional stretch (Phase 2, if time permits)

- LLM **query reformulation** before retrieval (compare with/without in UI demo).
- Simple **faithfulness check**: answer sentence ↔ cited passage overlap score.
- Batch demo script exporting Q/A pairs for manual review.

---

## 3. Architecture

### High-level flow

```
┌──────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Streamlit│───►│ search_pipeline │───►│ retrieval (8002) │───►│ ranked doc_ids  │
│  app_ui  │    │  (+ refine 8003)│    │                  │    └────────┬────────┘
└──────────┘    │  (+ pers. 8004) │    └──────────────────┘             │
                └────────┬────────┘                                      │
                         │ if use_rag                                     ▼
                         ▼                                    ┌─────────────────────┐
                ┌─────────────────┐                           │ MySQL documents     │
                │ rag_service     │◄── fetch passage texts ──│ (id, content)       │
                │ (8005)          │                           └─────────────────────┘
                │  Gemini API     │
                └────────┬────────┘
                         ▼
                { answer, citations, context_used, timing }
```

### SOA alignment

Follow the same pattern as `personalization_service` (Task 16):

- Independent service, own port, own `app/models.py`.
- Shared config in `shared/ir_config.py`.
- Orchestration in `shared/search_pipeline.py`.
- UI calls orchestrator, not Gemini directly (keeps API key server-side).

### Service ports (proposed)

| Service | Port | Role |
|---------|------|------|
| preprocessing | 8000 | Tokenization |
| retrieval | 8002 | Ranked search |
| query_refinement | 8003 | Optional query improve |
| personalization | 8004 | Optional rerank |
| **rag** | **8005** | **Context + Gemini answer** |

---

## 4. Component design

### 4.1 `rag_service/` layout

```
rag_service/
  app/
    __init__.py
    main.py                 # FastAPI: /generate, /health
    models.py               # Pydantic request/response
    core/
      __init__.py
      context_builder.py    # doc_ids → formatted context blocks
      doc_fetcher.py        # MySQL lookup (reuse pattern from personalization)
      gemini_client.py      # google-generativeai wrapper
      prompts.py            # system + user prompt templates
      token_budget.py       # truncate context to fit model window
```

### 4.2 API contract

#### `POST /generate`

**Request**

```json
{
  "query": "how does photosynthesis work",
  "results": { "doc_123": 12.4, "doc_456": 9.1 },
  "top_context_docs": 5,
  "max_context_chars": 12000,
  "model": "gemini-2.0-flash",
  "include_citations": true
}
```

`results` is the same `doc_id → score` map returned by retrieval (optionally reranked).

**Response**

```json
{
  "status": "success",
  "query": "how does photosynthesis work",
  "answer": "Photosynthesis converts light energy into chemical energy [doc_123]...",
  "citations": [
    { "doc_id": "doc_123", "snippet": "...", "retrieval_score": 12.4 }
  ],
  "context_doc_ids": ["doc_123", "doc_456"],
  "missing_doc_ids": [],
  "model": "gemini-2.0-flash",
  "timing": { "fetch_ms": 8.2, "generate_ms": 1200.5, "total_ms": 1215.0 }
}
```

#### `GET /health`

```json
{
  "status": "healthy",
  "service": "rag_service",
  "gemini_configured": true,
  "database_connected": true,
  "default_model": "gemini-2.0-flash"
}
```

### 4.3 Context builder

1. Take top `top_context_docs` from ranked results (default **5**).
2. `SELECT id, content FROM documents WHERE id IN (...)` — same SQL pattern as `personalization_service/app/core/doc_terms.py`.
3. Format each passage:

   ```
   [DOC doc_123 score=12.40]
   <passage text, max ~2000 chars per doc>
   ```

4. Apply **global char budget** (`max_context_chars`, default 12K) — trim lowest-scored docs first.
5. Track `missing_doc_ids` when MySQL has no row (index built but DB not migrated).

**Fallback:** If no passages found, return a structured error; UI shows “لا توجد نصوص للوثائق المسترجعة — شغّل migrate_to_db.py”.

### 4.4 Gemini client

**Dependency:** `google-generativeai` (add to `requirements.txt`).

**Environment:**

```powershell
$env:GEMINI_API_KEY="your_key_here"
# optional overrides:
$env:IR_RAG_MODEL="gemini-2.0-flash"
$env:IR_RAG_URL="http://127.0.0.1:8005"
```

**Client behavior:**

- Configure once at startup: `genai.configure(api_key=...)`.
- Use `GenerativeModel(model_name).generate_content(...)`.
- Set `generation_config`: `temperature=0.2`, `max_output_tokens=1024` (low temperature for faithfulness).
- Handle API errors → HTTP 502 with message; never log the API key.
- **No key in git** — document in developer guide; add `GEMINI_API_KEY` to `.gitignore` patterns if using `.env`.

**Recommended model:** `gemini-2.0-flash` (fast, cost-effective for demos). Allow override via request or `IR_RAG_MODEL`.

### 4.5 Prompt template (`prompts.py`)

**System instruction (English — MS MARCO is English):**

```
You are a helpful search assistant. Answer the user's question using ONLY the
provided passage excerpts. If the passages do not contain enough information,
say you cannot answer confidently from the sources. Cite supporting passages
as [DOC <doc_id>] after each claim. Do not invent facts not present in the passages.
```

**User message:**

```
Question: {query}

Retrieved passages:
{context_blocks}

Answer in clear natural language with citations.
```

This keeps generation **grounded** and auditable for the report.

### 4.6 Pipeline extension (`shared/search_pipeline.py`)

Add:

```python
def search_with_rag(
    ...,
    use_rag: bool = False,
    rag_top_context_docs: int = 5,
    rag_timeout: int = 60,
) -> Dict[str, Any]:
```

Flow:

1. Call existing `search_with_personalization(...)` (or `search_with_optional_refinement` if personalization off).
2. If `use_rag` and search succeeded with results:
   - `POST {RAG_URL}/generate` with `query=raw_query`, `results=search["results"]`.
3. Return `{ search, refinement, personalization, rag }`.

Config additions in `shared/ir_config.py`:

```python
RAG_URL = os.environ.get("IR_RAG_URL", "http://127.0.0.1:8005")
RAG_DEFAULT_MODEL = os.environ.get("IR_RAG_MODEL", "gemini-2.0-flash")
RAG_TOP_CONTEXT_DOCS = int(os.environ.get("IR_RAG_TOP_CONTEXT_DOCS", "5"))
RAG_MAX_CONTEXT_CHARS = int(os.environ.get("IR_RAG_MAX_CONTEXT_CHARS", "12000"))
```

Helper: `rag_generate_url()`.

---

## 5. UI changes (`app_ui.py`)

### Sidebar

- Checkbox: **إجابة ذكية (RAG)** — disabled if RAG health check fails.
- Sub-option: number of context passages (3 / 5 / 8).
- Caption: requires MySQL documents + `rag_service` + `GEMINI_API_KEY`.

### Results area

When RAG enabled:

1. **Answer card** (top) — generated text with citation badges linking to result rows.
2. **Ranked list** (below) — unchanged doc_id + score table (assignment still needs ranked output).
3. **Expandable trace** — context snippets sent to Gemini, timing, model name.

Execution mode mapping (assignment §9):

| UI mode | Retrieval | Refinement | Personalization | RAG |
|---------|-----------|------------|-----------------|-----|
| Basic only | ✓ | ✗ | ✗ | ✗ |
| Basic + additional | ✓ | optional | optional | optional |

RAG is an **additional feature**, not a replacement for basic search.

---

## 6. Implementation phases

### Phase 0 — Prerequisites (½ day)

- [ ] Confirm MySQL `documents` populated: `python migrate_to_db.py`
- [ ] Obtain Gemini API key; test with minimal script
- [ ] Add `google-generativeai` to `requirements.txt`
- [ ] Document env vars in `docs/developer-guide.md`

### Phase 1 — Service skeleton (1 day)

- [ ] Create `rag_service/app/main.py` with `/health` (gemini + db checks)
- [ ] `doc_fetcher.py` — extract shared `fetch_document_texts` to `shared/doc_store.py` OR duplicate thin wrapper (prefer **shared** to avoid drift with personalization)
- [ ] `context_builder.py` + unit tests
- [ ] Register `IR_RAG_URL` in `shared/ir_config.py`

### Phase 2 — Gemini integration (1 day)

- [ ] `gemini_client.py` with error handling
- [ ] `prompts.py` + `token_budget.py`
- [ ] `POST /generate` end-to-end
- [ ] Tests with `unittest.mock` patching `generate_content`

### Phase 3 — Pipeline + UI (1 day)

- [ ] `search_with_rag()` in `shared/search_pipeline.py`
- [ ] Streamlit toggle + answer panel + health check
- [ ] Manual smoke: query → answer with citations

### Phase 4 — Evaluation & report (1 day)

RAG is **not** evaluated with MAP/P@10 on doc rankings. For the report:

| Evaluation type | Method |
|-----------------|--------|
| Retrieval unchanged | Existing `evaluation_service` — RAG off |
| RAG qualitative | 10–20 dev queries: manual review of faithfulness |
| Optional automated | `scripts/run_rag_demo.py` — export JSON with query, answer, citations |
| Latency | Record `timing` from `/generate`; compare retrieval-only vs RAG |

Report section (Arabic):

- Architecture diagram including RAG service
- Why RAG is separate from ranking metrics
- Example Q/A with citations
- Limitations: hallucination risk, English-only, API latency/cost

---

## 7. Shared code refactor (recommended)

Today `personalization_service/app/core/doc_terms.py` owns `fetch_document_texts`. For RAG:

**Move to** `shared/doc_store.py`:

```python
def fetch_document_texts(doc_ids: List[str]) -> Dict[str, str]: ...
```

Update personalization imports. Single source for passage lookup.

---

## 8. Testing strategy

| Test file | Coverage |
|-----------|----------|
| `tests/test_rag_context.py` | Context ordering, truncation, missing docs |
| `tests/test_rag_api.py` | FastAPI `/generate` with mocked Gemini |
| `tests/test_rag_pipeline.py` | `search_with_rag` orchestration (mock HTTP) |

No live Gemini calls in CI — mock only.

Manual test checklist:

```powershell
# Terminal 1–4: preprocessing, retrieval, (refinement), (personalization)
cd rag_service
uvicorn app.main:app --host 127.0.0.1 --port 8005 --reload

$env:GEMINI_API_KEY="..."
streamlit run app_ui.py
```

Query: `how to tie a tie` → expect answer citing MS MARCO passages.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Hallucination | Low temperature, strict system prompt, require citations |
| Missing passage text | Health check `database_connected`; clear UI error |
| API cost / rate limits | Default `gemini-2.0-flash`; cap `top_context_docs` and chars |
| Latency (+1–3s) | Show spinner; RAG optional toggle |
| Key leakage | Env var only; never commit; health returns `gemini_configured: bool` not key |
| Index/DB ID mismatch | Log `missing_doc_ids`; document migrate step in guide |

---

## 10. How this maps to “RAG domain”

In IR terms, our RAG stack is:

| RAG stage | Our implementation |
|-----------|-------------------|
| **Indexing** | Existing `indexing_service` (BM25, embeddings, FAISS) — no change |
| **Retrieval** | `retrieval_service` + optional refine/personalize |
| **Augmentation** | `context_builder` loads passage texts, formats prompt |
| **Generation** | Gemini via `gemini_client` |

We use **lexical + dense hybrid retrieval** already built in the project rather than a separate RAG vector store. That is valid RAG: retrieval can be any IR system; generation is the augmentation layer.

Deferred from Task 5 (**LLM query rewriting**) can be added as `POST /rewrite` in the same service later, but keep v1 focused on **answer generation** to reduce scope.

---

## 11. File checklist (new / modified)

### New

- `rag_service/app/main.py`
- `rag_service/app/models.py`
- `rag_service/app/core/context_builder.py`
- `rag_service/app/core/gemini_client.py`
- `rag_service/app/core/prompts.py`
- `rag_service/app/core/token_budget.py`
- `shared/doc_store.py` (optional refactor)
- `tests/test_rag_*.py`
- `scripts/run_rag_demo.py` (optional)

### Modified

- `shared/ir_config.py` — RAG URLs and defaults
- `shared/search_pipeline.py` — `search_with_rag()`
- `app_ui.py` — RAG toggle + answer panel
- `requirements.txt` — `google-generativeai`
- `docs/developer-guide.md` — run instructions
- `.gitignore` — `.env` if used

---

## 12. Quick start (after implementation)

```powershell
# One-time
pip install google-generativeai
$env:GEMINI_API_KEY="your_key"
python migrate_to_db.py   # if not done

# Services
cd rag_service
uvicorn app.main:app --host 127.0.0.1 --port 8005 --reload

# UI: enable "إجابة ذكية (RAG)"
streamlit run app_ui.py
```

**Direct API test:**

```powershell
curl -X POST http://127.0.0.1:8005/generate `
  -H "Content-Type: application/json" `
  -d '{"query":"how to tie a tie","results":{"123":9.5,"456":8.1},"top_context_docs":3}'
```

---

## 13. Success criteria

1. User can search with RAG off → same behavior and metrics as today.
2. User can enable RAG → natural-language answer with `[DOC id]` citations.
3. Ranked doc list still visible (IR core preserved).
4. Service runs independently on port 8005 with `/health`.
5. Tests pass without real API key.
6. Arabic report section explains design, demo, and limitations.

---

## 14. Estimated effort

| Phase | Effort |
|-------|--------|
| Phase 0–1 | 1–1.5 days |
| Phase 2 | 1 day |
| Phase 3 | 1 day |
| Phase 4 | 0.5–1 day |
| **Total** | **~3.5–4.5 days** (one developer) |

This fits as a single additional feature (§10) alongside existing Task 16 personalization for a 6–7 member team.
