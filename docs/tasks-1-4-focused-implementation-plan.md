# Tasks 1-4 Focused Implementation and Refactoring Plan

## Short Answer: Should You Use 200K Now?
Use a **progressive scale strategy**, not full 200K from day one.

- During daily development, keep a **small subset** (for fast iteration).
- Before declaring each task "done", test on a **medium subset**.
- Before final submission/evaluation, run a **full-scale indexing/evaluation pass** (target assignment scale).

This is the standard IR engineering approach: fast dev loop + staged scale validation.

## Recommended Dataset Scale by Stage
- **Stage A (Dev Loop)**: 2K-10K docs (very fast debugging).
- **Stage B (Pre-validation)**: 20K-50K docs (performance and stability check).
- **Stage C (Final Validation)**: full dataset scale (up to assignment target).

Rule: never ship logic that was tested only on tiny toy data.

---

## Current Gaps to Fix (Tasks 1-4)
1. VSM cosine similarity is incomplete (query norm used, document norm missing).
2. Serial hybrid uses too small candidate pool (`top_n_filter=2`), hurting recall.
3. Index storage paths are duplicated (`index_data` vs `indexing_service/index_data`).
4. No reproducible indexing manifest/config snapshot.
5. No systematic evaluation pipeline (MAP, Recall, P@10, nDCG) connected to retrieval outputs.
6. Runtime and scalability risks:
   - full-scan embedding search for all docs,
   - engine reload per request,
   - JSON-only storage for large scale.

---

## Implementation Plan (Concrete Steps)

## Phase 0 - Baseline Freeze and Config Cleanup
### Goal
Make the current baseline reproducible before changing logic.

### Steps
1. Add centralized config (single source for paths/ports/modes).
2. Enforce one canonical index output directory.
3. Add run metadata file for each index build:
   - dataset name
   - document count indexed
   - preprocessing flags
   - embedding model name
   - timestamp
   - git commit hash (optional but recommended)

### Files to Touch
- `indexing_service/app/core/indexer.py`
- `indexing_service/app/main.py`
- `retrieval_service/app/main.py`

### Done Criteria
- Running indexing twice with same config produces consistent artifact structure and traceable metadata.

---

## Phase 1 - Task 1 Hardening (Data Pre-Processing)
### Goal
Guarantee consistent preprocessing behavior for docs and queries.

### Steps
1. Lock preprocessing defaults in one place (avoid drift across services).
2. Add explicit validation tests for edge cases:
   - empty text
   - URL-only text
   - punctuation/noise
   - numbers-heavy text
3. Log preprocessing mode in index metadata.
4. Ensure fallback behavior is explicit if spaCy model is missing.

### Files to Touch
- `preprocessing_service/app/core/cleaner.py`
- `preprocessing_service/app/main.py`
- index metadata write logic in indexing service

### Done Criteria
- Same input text gives predictable token output across runs/environments (or clearly documented fallback).

---

## Phase 2 - Task 2 Fixes (Representations)
### Goal
Improve retrieval correctness and hybrid quality.

### Steps
1. Fix VSM to true cosine:
   - compute/store document vector norm
   - divide by `(query_norm * doc_norm)`
2. Increase serial hybrid candidate pool:
   - replace fixed `top_n_filter=2` with configurable default (start at 100).
3. Keep parallel hybrid RRF, but optimize rank lookup complexity.
4. Add representation-level comparison hooks for evaluation service.

### Files to Touch
- `retrieval_service/app/main.py` (VSM scoring path)
- `retrieval_service/app/core/search_engine.py` (hybrid logic)
- metadata/index generation path for VSM norms

### Done Criteria
- VSM and hybrid outputs are stable and materially better on medium-scale test set.

---

## Phase 3 - Task 3 Hardening (Indexing)
### Goal
Make indexing robust and scale-aware.

### Steps
1. Parameterize indexing scale:
   - dev mode doc cap
   - pre-validation cap
   - final full run mode
2. Add post-index sanity checks:
   - indexed docs count
   - empty-token docs count
   - missing embeddings count
3. Ensure retrieval service reads exactly the same index directory built by indexing.
4. Keep JSON for now, but design adapter boundary for future storage backend (FAISS/SQLite/etc).

### Files to Touch
- `indexing_service/app/core/indexer.py`
- `indexing_service/app/main.py`
- `retrieval_service/app/main.py`

### Done Criteria
- One command/config can rebuild index reliably for any target scale stage.

---

## Phase 4 - Task 4 Hardening (Query Processing)
### Goal
Make query flow correct, efficient, and testable.

### Steps
1. Stop reinitializing engines on every request unless index changed.
2. Add explicit mode validation and error messages.
3. Add per-query timing logs (preprocess time, ranking time, total latency).
4. Add query regression tests (known query -> expected top results pattern).

### Files to Touch
- `retrieval_service/app/main.py`
- `retrieval_service/app/core/search_engine.py`
- optional test scripts

### Done Criteria
- Query pipeline is deterministic, observable, and latency is acceptable on Stage B scale.

---

## Phase 5 - Evaluation Backbone (Needed to Confirm Tasks 1-4)
### Goal
Prove quality, not just functionality.

### Steps
1. Implement evaluation runner using test queries + qrels.
2. Report at minimum:
   - MAP
   - Recall
   - Precision@10
   - nDCG
3. Run metrics per representation mode:
   - VSM
   - BM25
   - Embedding
   - Hybrid parallel
   - Hybrid serial
4. Compare Stage B vs Stage C results and record trade-offs.

### Done Criteria
- You can show metric tables and explain why your chosen defaults are justified.

---

## "Right Call" Decision Framework for Your Team
- If you are changing core logic daily -> stay on Stage A.
- If you finished a phase and want confidence -> run Stage B.
- If you are preparing submission/demo -> run Stage C.

Minimum policy:
- No merge to main branch for retrieval logic unless it passed Stage A + Stage B checks.
- No "task complete" claim unless metrics were produced on evaluation queries.

---

## Immediate Next Actions (In Order)
1. Unify index path and config source.
2. Fix VSM cosine normalization.
3. Raise serial hybrid candidate depth and make it configurable.
4. Add indexing manifest and sanity checks.
5. Build evaluation runner for MAP/Recall/P@10/nDCG.
6. Execute staged dataset runs (A -> B -> C).

If you follow these six actions, you will have a clear and defensible implementation status for Tasks 1-4 and be ready to continue confidently.
