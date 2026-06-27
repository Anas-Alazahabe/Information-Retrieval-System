# Task 16 - Personalization

## Requirement Description

Additional feature #16: adapt search results to individual users using persistent profiles built from queries and clicks, with post-retrieval re-ranking (not query expansion).

Distinct from Task 5 `history` refinement (session-only query term merge in `query_refinement_service`).

## What Was Implemented

### Personalization service (`personalization_service`, port 8004)

| Endpoint | Purpose |
|----------|---------|
| `POST /events/query` | Log query + update MySQL profile |
| `POST /events/click` | Log click + update profile (2x weight) |
| `GET /profile/{user_id}` | Profile summary |
| `POST /personalize/rerank` | Re-rank retrieval results |
| `DELETE /profile/{user_id}` | Reset user |
| `GET /health` | Service + MySQL status |

### MySQL schema (`scripts/init_personalization_schema.py`)

- `users`, `user_query_events`, `user_click_events`, `user_interest_terms`
- Reuses `documents` table from `migrate_to_db.py` (~200K MS MARCO passages)

### Re-ranking formula

```
final = alpha * norm(base_score) + (1 - alpha) * norm(profile_overlap_score)
```

Default `alpha = 0.7` (`IR_PERSONALIZATION_ALPHA`).

Profile overlap: sum of interest term weights present in document text (from MySQL `documents.content`).

### Pipeline (`shared/search_pipeline.py`)

`search_with_personalization()` orchestrates: optional refine → search (pool=100) → rerank → log query event.

### UI (`app_ui.py`)

- Sidebar toggle: تخصيص النتائج
- Users: `demo_health`, `demo_tech`, custom
- Click logging per result row
- Personalization trace in search details expander

### Scripts

- `scripts/seed_demo_profiles.py` — seed demo users
- `scripts/run_personalization_eval.py` — simulated-user eval (baseline vs personalized)

## Relevant Files

- `personalization_service/app/main.py`
- `personalization_service/app/core/reranker.py`
- `personalization_service/app/core/profile_store.py`
- `shared/db_config.py`
- `shared/search_pipeline.py`
- `tests/test_personalization_*.py`

## Evaluation

```powershell
python scripts/run_personalization_eval.py --scale full --max-queries 50 --modes bm25,embedding
```

Requires: preprocessing (8000), retrieval (8002), personalization (8004), MySQL with documents + schema.

Outputs under `evaluation_results/`:

- `eval_personalization_baseline_full_*.json`
- `eval_personalization_simulated_full_*.json`
- `personalization_ablation_summary_full_*.json`

## Limitations

- Simulated oracle clicks on qrels-relevant docs (upper bound on gains).
- Re-ranking needs document text in MySQL; missing docs get zero profile boost.
- Task 5 session `history` is separate and not counted as personalization.

## IR Quality Assessment

- **SOA**: Independent service; retrieval unchanged.
- **Persistence**: Cross-session profiles in MySQL.
- **Demonstrability**: Same query, different users → different rankings after seeding.
