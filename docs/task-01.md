# Task 01 - Data Pre-Processing

## Requirement Description
Assignment Task 1 requires preprocessing raw documents using suitable IR text-cleaning techniques (e.g., normalization, stopword removal, stemming, lemmatization) before indexing and retrieval.

## What Was Implemented
- A dedicated FastAPI microservice in `preprocessing_service`.
- Endpoints:
  - `/preprocess` for single query/text.
  - `/preprocess-batch` for document batches.
- Cleaner pipeline in `preprocessing_service/app/core/cleaner.py`:
  - URL removal.
  - Lowercasing and whitespace normalization.
  - Two modes:
    - Lemmatization mode (spaCy, if available).
    - Traditional mode (regex/punctuation stripping + split).
  - Optional stopword removal (NLTK English stopwords).
  - Optional stemming (Porter).

## Relevant Files and Components
- `preprocessing_service/app/main.py`
- `preprocessing_service/app/core/cleaner.py`
- Integrations:
  - Called by indexing path (`indexing_service/app/core/indexer.py`).
  - Called by retrieval path (`retrieval_service/app/main.py`).

## Algorithms and Techniques
- Rule-based normalization/token filtering.
- NLTK stopword filtering.
- Porter stemming.
- spaCy lemmatization fallback logic.

## Inputs and Outputs
- Input: raw text(s), flags (`use_stemming`, `use_lemmatization`, `remove_stopwords`).
- Output: token list(s) and counts/batch size.

## IR Quality Assessment
- **Dataset size appropriateness**: preprocessing approach is acceptable for large corpora, but current run artifacts show only tiny datasets.
- **Split/preparation correctness**: document/query preprocessing consistency is mostly respected (same service used in both indexing and query time).
- **Algorithm suitability**: suitable baseline for lexical IR; semantic-heavy tasks may need stronger normalization or domain adaptation.
- **Metric relevance/calculation**: no direct metrics here; preprocessing impact is not quantified in current project.
- **Methodological risks**:
  - English-only stopwords/lemmatizer can fail on multilingual or mixed-language data.
  - spaCy model optional behavior can cause environment-dependent tokenization drift.
  - No explicit handling for stemming+lemmatization strategy beyond sequential application.
- **IR best-practice compliance**: fair baseline, but missing reproducibility controls (fixed preprocessing config logs/versioning).

## Observations and Recommendations
- Add explicit preprocessing configuration persistence in metadata (language, options, model versions).
- Validate/token-audit samples from both datasets before full indexing.
- Add deterministic test cases for edge text (numbers, punctuation, empty docs, URLs, Unicode).
- Consider query/document normalization parity checks in evaluation scripts.
