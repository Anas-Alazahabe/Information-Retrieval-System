# Task 6 — Query Matching & Ranking (condensed plan)

See full implementation in `docs/task-06.md`.

## Sessions
- **Session A**: Matcher module, registry, tests, API metadata
- **Session B**: FAISS, evaluation output, UI, documentation

## Matcher map
| Mode | Method |
|------|--------|
| vsm | cosine_similarity |
| bm25 | bm25 |
| embedding | cosine_similarity |
| hybrid_parallel | rrf |
| hybrid_serial | bm25_filter_cosine_rerank |

## Key commands
```powershell
# Run matcher tests
python -m pytest tests/test_matchers.py -q

# Evaluate all modes
python -m evaluation_service.run --scale preval --max-queries 100 --output-dir evaluation_results
```
