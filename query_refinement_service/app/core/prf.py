"""Pseudo-relevance feedback (simplified RM3) using BM25 first-pass retrieval."""

import logging
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import requests

from shared.index_store import JsonIndexStore
from shared.ir_config import (
    INDEX_DIR,
    PRF_ORIGINAL_QUERY_WEIGHT,
    PRF_TOP_K_DOCS,
    PRF_TOP_M_TERMS,
    RETRIEVAL_URL,
)

logger = logging.getLogger(__name__)

PRF_TIMEOUT_SECONDS = 10

# Loading bm25_index.json on every /refine call OOMs at 200K scale; cache per process.
_BM25_FEEDBACK_CACHE: Dict[str, Dict] = {}


def _get_bm25_feedback_data(index_dir: str) -> Tuple[Dict, Dict]:
    """Load BM25 postings and doc lengths once, reusing cache when index files are unchanged."""
    bm25_path = os.path.join(index_dir, "bm25_index.json")
    meta_path = os.path.join(index_dir, "metadata.json")
    if not os.path.exists(bm25_path):
        return {}, {}

    bm25_mtime = os.path.getmtime(bm25_path)
    meta_mtime = os.path.getmtime(meta_path) if os.path.exists(meta_path) else 0.0
    cache_key_mtime = max(bm25_mtime, meta_mtime)

    cached = _BM25_FEEDBACK_CACHE.get(index_dir)
    if cached and cached.get("mtime") == cache_key_mtime:
        return cached["bm25_index"], cached["doc_lengths"]

    store = JsonIndexStore(index_dir)
    try:
        bm25_index = store.load_bm25()
        metadata = store.load_metadata()
    except MemoryError:
        logger.warning("PRF skipped: BM25 index too large to load into memory")
        return {}, {}
    except OSError as exc:
        logger.warning("PRF skipped: could not read BM25 index: %s", exc)
        return {}, {}

    doc_lengths = metadata.get("doc_lengths", {})
    _BM25_FEEDBACK_CACHE[index_dir] = {
        "mtime": cache_key_mtime,
        "bm25_index": bm25_index,
        "doc_lengths": doc_lengths,
    }
    return bm25_index, doc_lengths
SHORT_QUERY_MAX_TERMS = 5
_VALID_FEEDBACK_TERM = re.compile(r"^[a-z][a-z0-9-]*$")


def _is_valid_feedback_term(term: str) -> bool:
    """Accept lowercase ASCII index terms; reject mojibake and punctuation artifacts."""
    if not term or len(term) < 2:
        return False
    return bool(_VALID_FEEDBACK_TERM.match(term))


def collect_doc_term_weights(doc_id: str, bm25_index: Dict) -> Dict[str, int]:
    """Extract term frequencies for a document from BM25 postings."""
    term_weights: Dict[str, int] = {}
    for term, postings in bm25_index.items():
        if doc_id in postings:
            term_weights[term] = int(postings[doc_id].get("tf", 0))
    return term_weights


def _first_pass_retrieval(
    query_text: str,
    top_k: int,
    retrieval_url: str,
    preserve_wh_words: bool = False,
) -> Dict[str, float]:
    """Run BM25 first-pass retrieval via the retrieval service."""
    search_url = f"{retrieval_url.rstrip('/')}/search"
    response = requests.post(
        search_url,
        json={
            "query": query_text,
            "representation_mode": "bm25",
            "top_k": top_k,
            "preserve_wh_words": preserve_wh_words,
        },
        timeout=PRF_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", {})
    return {str(doc_id): float(score) for doc_id, score in results.items()}


def _score_feedback_terms(
    feedback_docs: Dict[str, float],
    bm25_index: Dict,
    query_tokens: List[str],
    doc_lengths: Dict,
) -> Tuple[List[Tuple[str, float]], int]:
    """Score candidate feedback terms from top retrieved documents."""
    query_set = {t.lower() for t in query_tokens}
    feedback_weights: Counter = Counter()
    filtered_count = 0

    for doc_id, doc_score in feedback_docs.items():
        if doc_score <= 0:
            continue
        term_tfs = collect_doc_term_weights(doc_id, bm25_index)
        if not term_tfs:
            continue
        doc_len = doc_lengths.get(doc_id) or doc_lengths.get(str(doc_id))
        if not doc_len:
            doc_len = sum(term_tfs.values()) or 1
        for term, tf in term_tfs.items():
            if term in query_set:
                continue
            if not _is_valid_feedback_term(term):
                filtered_count += 1
                continue
            feedback_weights[term] += doc_score * (tf / doc_len)

    ranked = sorted(feedback_weights.items(), key=lambda x: (-x[1], x[0]))
    return ranked, filtered_count


def prf_rm3(
    query_tokens: List[str],
    top_k_docs: int = PRF_TOP_K_DOCS,
    top_m_terms: int = PRF_TOP_M_TERMS,
    original_weight: float = PRF_ORIGINAL_QUERY_WEIGHT,
    retrieval_url: str = RETRIEVAL_URL,
    index_dir: str = INDEX_DIR,
    preserve_wh_words: bool = False,
) -> Tuple[List[str], str]:
    """Two-pass PRF: BM25 retrieval then feedback term extraction from index postings."""
    del original_weight  # reserved for weighted blending in future tuning

    if not query_tokens:
        return [], "PRF skipped: empty query tokens."

    short_query_cap = len(query_tokens) <= 2
    effective_top_m = min(top_m_terms, SHORT_QUERY_MAX_TERMS) if short_query_cap else top_m_terms

    query_text = " ".join(query_tokens)

    try:
        feedback_docs = _first_pass_retrieval(
            query_text=query_text,
            top_k=top_k_docs,
            retrieval_url=retrieval_url,
            preserve_wh_words=preserve_wh_words,
        )
    except Exception as exc:
        logger.warning("PRF first pass failed: %s", exc)
        return [], "PRF skipped: retrieval unavailable."

    if not feedback_docs:
        return [], "PRF skipped: no first-pass results."

    bm25_index, doc_lengths = _get_bm25_feedback_data(index_dir)
    if not bm25_index:
        return [], "PRF skipped: BM25 index not available."

    ranked, filtered_count = _score_feedback_terms(
        feedback_docs, bm25_index, query_tokens, doc_lengths
    )
    if not ranked:
        top_ids = ", ".join(list(feedback_docs.keys())[:5])
        return [], f"PRF skipped: no expandable terms from feedback docs ({top_ids})."

    feedback_terms = [term for term, _ in ranked[:effective_top_m]]
    top_ids = ", ".join(list(feedback_docs.keys())[:5])
    terms_str = ", ".join(feedback_terms[:10])

    notes = []
    if filtered_count:
        notes.append(f"{filtered_count} filtered")
    if short_query_cap:
        notes.append("short-query cap")
    note_suffix = f" ({', '.join(notes)})" if notes else ""

    explanation = (
        f"PRF from top-{len(feedback_docs)} BM25 docs ({top_ids}); "
        f"added terms: {terms_str}{note_suffix}."
    )
    return feedback_terms, explanation
