# Project Notes - Critical Assessment and Task 5 Readiness

## Missing or Incomplete Implementations
- Original assignment text requires two large datasets with test/qrels, but the supervisor approved using one dataset; current implementation should still document this exception explicitly in the final report.
- Evaluation requirements (MAP, Recall, P@10, nDCG before/after extra features) are not implemented in the current code path.
- `evaluation_service` and `api_gateway` directories exist but appear largely unimplemented.
- UI does not expose dataset selection despite assignment requirement for dataset choice via interface.

## Technical Debt
- Duplicate index artifact roots (`index_data` and `indexing_service/index_data`) create operational ambiguity.
- Hardcoded service URLs and ports reduce deployment flexibility.
- No dependency lockfile or complete requirements manifests for each service.
- No integration tests for cross-service contracts or ranking consistency.

## Potential Bugs / Design Issues
- `retrieval_service` VSM score uses query norm but not document norm (not full cosine similarity). *
- `HybridSearchEngine.search_parallel` uses `list.index()` repeatedly, causing avoidable O(n^2) ranking overhead. *
- Serial hybrid default `top_n_filter=2` is too restrictive and likely harms recall. *
- Full-scan embedding similarity over all documents is expensive for large collections.
- Runtime behavior may differ by environment if spaCy model is unavailable.

## Deviations from Assignment Requirements
- Dataset scale requirement (>200K per dataset) is not reflected in checked artifacts. *
- Single-dataset usage is not treated as a deviation because it is supervisor-approved; this must be clearly stated in deliverables to avoid grading ambiguity.
- Formal IR metric evaluation pipeline is missing.
- SOA intent exists, but complete independent services list from assignment is not fully realized yet.

## Scalability, Performance, and Maintainability Concerns
- JSON file storage/loading is not ideal for large-scale postings and vectors.
- Reinitializing retrieval engines on each query adds avoidable latency.
- Lack of standardized configuration management (env vars/settings classes).
- No index versioning or manifest to ensure reproducible experiments.

## Per-Task Methodological Checks (Tasks 1-4)

### Task 1 (Preprocessing)
- Dataset size appropriate? **Partially**; pipeline is light enough, but execution evidence is small-scale only.
- Split/preparation correct? **Mostly**, for query/doc preprocessing consistency.
- Algorithms/evaluation suitable? **Algorithms yes**; evaluation impact not measured.
- Metrics meaningful/correct? **Not applicable directly**, but no ablation metrics exist.
- Leakage/overfitting risks? **Low direct leakage**, moderate risk of inconsistent behavior across environments.
- IR best-practice alignment? **Baseline aligned**, needs reproducibility controls.

### Task 2 (Representation)
- Dataset size appropriate? **Potentially**, but current brute-force embedding retrieval not scalable.
- Split/preparation correct? **Incomplete**, no robust experimental split setup shown.
- Algorithms/evaluation suitable? **Good baseline choices** (lexical + dense + hybrid).
- Metrics meaningful/correct? **Not yet measured**.
- Leakage/overfitting risks? **Low leakage evidence**, but weak validation process.
- IR best-practice alignment? **Partially aligned**, needs better calibration and evaluation.

### Task 3 (Indexing)
- Dataset size appropriate? **Current storage approach is weak** for required scale.
- Split/preparation correct? **Insufficient evidence** around full corpus coverage and qrels alignment.
- Algorithms/evaluation suitable? **Inverted/BM25 structures are suitable**, persistence layer is prototype-grade.
- Metrics meaningful/correct? **Missing indexing QA metrics**.
- Leakage/overfitting risks? **Low leakage**, but high risk of incomplete indexing due caps/path confusion.
- IR best-practice alignment? **Partially aligned**.

### Task 4 (Query Processing)
- Dataset size appropriate? **Partially**, but dense retrieval path is not optimized.
- Split/preparation correct? **Mostly consistent preprocessing**, but no benchmark query protocol.
- Algorithms/evaluation suitable? **Suitable baseline**, missing rigorous evaluation.
- Metrics meaningful/correct? **Not integrated**.
- Leakage/overfitting risks? **No clear leakage**, but methodological weakness from absent test harness.
- IR best-practice alignment? **Moderate**; correctness/performance gaps remain.

## Task 5 Readiness Assessment

### What Is Working Well
- Clear microservice decomposition for preprocessing, indexing, retrieval.
- Multiple retrieval paradigms already implemented (VSM, BM25, embedding, hybrid).
- BM25 runtime parameter control exists in API and UI.
- Query and document preprocessing consistency is reasonably maintained.

### What Must Be Fixed Before Proceeding
- Establish a documented, supervisor-approved dataset setup (one dataset), and validate scale + qrels availability for that dataset.
- Implement evaluation pipeline (MAP, Recall, P@10, nDCG) and produce baseline reports.
- Resolve index path duplication and enforce a single source of truth.
- Correct VSM cosine formulation and revisit hybrid serial candidate size.

### Assumptions / Uncertainties
- Assumed current code snapshot reflects "completed first four tasks"; unseen files/notebooks may exist outside scanned paths.
- Assumed index artifacts in repo are representative; they appear toy/demo and may not reflect latest full runs.
- No runtime validation was executed in this review, so dependency/runtime issues remain unverified.

### Confidence to Continue to Task 5
- **Current status: partially ready, not fully production-ready for Task 5.**
- Development can continue if Task 5 is exploratory, but for assignment-grade progression the fixes above are strongly recommended first.
- After metric pipeline + dataset compliance + indexing/ranking corrections, confidence to proceed becomes high.
