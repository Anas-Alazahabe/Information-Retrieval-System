# Task 05 - Query Refinement

## Requirement Description

Task 5 adds a query refinement layer before retrieval to improve search quality on MS MARCO Passage. The assignment requires query formulation assistance (suggestions, session history) and measurable before/after evaluation (§8) across all representation modes.

## What Was Implemented

### Refinement service (`query_refinement_service`, port 8003)

- **`POST /refine`** — orchestrates enabled techniques and returns refined query + trace metadata.
- **`GET /suggest`** — prefix autocomplete over MS MARCO dev queries (UX-only).
- **`GET /health`** — service status + suggestion index status.

### Refinement techniques

| Technique | Priority | Evaluable on qrels? | Role |
|-----------|----------|---------------------|------|
| `query_preprocess` | P1 | Yes | Preserve WH-words in question-style queries |
| `synonyms` | P0 | Yes | WordNet expansion (bounded) |
| `prf` | P0 | Yes | BM25 first-pass pseudo-relevance feedback |
| `history` | P2 | No (demo) | Session history term merge with decay weights |
| Query suggestions | P2 | No (UX) | MS MARCO prefix autocomplete in UI |

### UI integration (`app_ui.py`)

- **Basic only** vs **Basic + additional features** execution modes (assignment §9).
- Refinement presets: Recommended stack, Preprocess only, Demo with history, Custom.
- Refinement trace panel (raw vs refined query, techniques, expanded terms).
- Session history stored in `st.session_state["search_history"]` (last 5 queries).
- Query suggestion chips below the input field.

### Evaluation integration

- `evaluation_service` supports `--use-refinement` and `--refinement-techniques`.
- `scripts/run_refinement_ablation.py` runs baseline + ablations + combined stack and writes summary JSON with deltas vs baseline.
- Shared orchestrator: `shared/search_pipeline.py` (`search_with_optional_refinement`).

## Relevant Files and Components

- `query_refinement_service/app/main.py` — FastAPI endpoints
- `query_refinement_service/app/core/refiner.py` — technique orchestration
- `query_refinement_service/app/core/history.py` — session history merge
- `query_refinement_service/app/core/prf.py` — PRF feedback terms
- `query_refinement_service/app/core/synonym_expander.py` — WordNet expansion
- `shared/search_pipeline.py` — refine-then-search orchestration
- `shared/query_suggestions.py` — prefix suggestion lookup
- `scripts/build_query_suggestion_index.py` — offline MS MARCO query index builder
- `scripts/run_refinement_ablation.py` — before/after ablation runner
- `evaluation_service/app/main.py` — metrics with optional refinement
- `evaluation_service/run.py` — CLI entry point
- `app_ui.py` — Streamlit UI

## Algorithms and Techniques

1. **Query-specific preprocessing** — detect WH-words / trailing `?`; set `preserve_wh_words` hint for preprocessing service.
2. **Session history** — lightweight tokenize prior queries; decay weights `1.0, 0.5, 0.25, …`; append up to 5 new terms.
3. **WordNet synonyms** — per-term synonym cap (`SYNONYM_MAX_PER_TERM=2`), total cap (`SYNONYM_MAX_TOTAL=8`).
4. **PRF (simplified RM3)** — BM25 first pass (`PRF_TOP_K_DOCS=10`); extract feedback terms from top-doc postings (`PRF_TOP_M_TERMS=15`).
5. **Query suggestions** — sorted MS MARCO query list + binary-search prefix match.

Technique order when all enabled: `query_preprocess → history → synonyms → prf`.

## Data Flow

```
User query
  → (optional) POST /refine  [query_refinement_service]
      → preprocessing service (tokenize)
      → retrieval service (PRF first pass only)
  → POST /search  [retrieval_service]
      → preprocessing service
      → rank (VSM / BM25 / embedding / hybrid)
  → results + refinement trace (UI)
```

## Parameter Choices

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `PRF_TOP_K_DOCS` | 10 | Enough feedback docs without noise on MS MARCO |
| `PRF_TOP_M_TERMS` | 15 | Bounded expansion; avoids query blow-up |
| `SYNONYM_MAX_PER_TERM` | 2 | Limits WordNet ambiguity |
| `SYNONYM_MAX_TOTAL` | 8 | Caps total static expansion |
| `HISTORY_MAX_QUERIES` | 5 | Short session window for demo |
| `HISTORY_MAX_TERMS` | 5 | Prevents history from dominating current query |

## Evaluation Results

Ablation run (dev index, BM25, 5 queries — pipeline validation):

- Report: `reports/refinement_ablation_summary_dev_20260618T090805Z.json`
- Result: **zero delta** on all metrics for this run because the 5K-doc dev index has limited qrels overlap with the first dev queries evaluated.

**For report-quality numbers**, run on Stage B+ index:

```powershell
python scripts/run_refinement_ablation.py --scale preval --max-queries 50
```

Prerequisites: preprocessing (8000), retrieval (8002), refinement (8003) running; index built with `--scale preval`.

Expected pattern on larger indexes: largest gains on **BM25** and **hybrid** modes; smaller delta on **embedding** (refinement expands lexical tokens fed to the encoder).

## IR Quality Assessment

- **Dataset fit**: PRF + synonym expansion target vocabulary mismatch between short MS MARCO queries and long passages — appropriate for this collection.
- **SOA separation**: Refinement stays upstream of retrieval; retrieval API unchanged.
- **Demo-only features**: `history` and query suggestions are excluded from batch qrels evaluation (no real user sessions in MS MARCO).
- **Evaluation hook**: Automated ablation script produces timestamped JSON under `reports/` for assignment §8 before/after comparison.
- **Honest limitation**: Gains depend on index scale and qrels coverage; dev-scale smoke tests may show zero delta even when the pipeline is correct.

## Observations and Recommendations

- Run final ablation on `--scale preval` or `--scale full` before the Arabic report.
- PRF adds latency (extra BM25 search per query); acceptable for demo, note in report.
- Consider corpus vocabulary filter for WordNet synonyms (optional enhancement).
- Use `docs/developer-guide.md` for service startup and eval commands.
