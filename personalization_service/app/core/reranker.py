"""Personalized re-ranking of retrieval results."""

from typing import Any, Dict, List, Tuple

from .doc_terms import fetch_document_texts
from .tokenizer import lightweight_tokenize
from shared.ir_config import PERSONALIZATION_ALPHA


def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return {doc_id: 1.0 if max_v > 0 else 0.0 for doc_id in scores}
    span = max_v - min_v
    return {doc_id: (score - min_v) / span for doc_id, score in scores.items()}


def _profile_scores(
    doc_ids: List[str],
    profile_terms: Dict[str, float],
    doc_texts: Dict[str, str],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for doc_id in doc_ids:
        content = doc_texts.get(doc_id, "")
        doc_tokens = set(lightweight_tokenize(content))
        score = sum(weight for term, weight in profile_terms.items() if term in doc_tokens)
        scores[doc_id] = score
    return scores


def rerank_results(
    results: Dict[str, float],
    profile_terms: Dict[str, float],
    *,
    alpha: float = PERSONALIZATION_ALPHA,
    query_text: str = "",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not results:
        return {}, {
            "personalization_applied": False,
            "alpha": alpha,
            "profile_terms_used": [],
            "boosted_docs": [],
            "explanation": "No retrieval results to re-rank.",
        }

    if not profile_terms:
        return dict(results), {
            "personalization_applied": False,
            "alpha": alpha,
            "profile_terms_used": [],
            "boosted_docs": [],
            "explanation": "User profile has no interest terms yet.",
        }

    doc_ids = list(results.keys())
    doc_texts = fetch_document_texts(doc_ids)
    missing_docs = [doc_id for doc_id in doc_ids if doc_id not in doc_texts]

    query_tokens = set(lightweight_tokenize(query_text))
    terms_used = [
        term
        for term, _ in sorted(profile_terms.items(), key=lambda item: (-item[1], item[0]))
        if term not in query_tokens
    ][:10]
    active_terms = {term: profile_terms[term] for term in terms_used if term in profile_terms}
    if not active_terms:
        active_terms = profile_terms

    profile_raw = _profile_scores(doc_ids, active_terms, doc_texts)
    norm_base = _min_max_normalize(results)
    norm_profile = _min_max_normalize(profile_raw)

    final_scores: Dict[str, float] = {}
    for doc_id in doc_ids:
        base = norm_base.get(doc_id, 0.0)
        prof = norm_profile.get(doc_id, 0.0)
        final_scores[doc_id] = alpha * base + (1.0 - alpha) * prof

    original_rank = {
        doc_id: rank for rank, doc_id in enumerate(
            sorted(results.keys(), key=lambda d: (-results[d], str(d)))
        )
    }
    reranked = dict(
        sorted(
            final_scores.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
    )
    new_rank = {doc_id: rank for rank, doc_id in enumerate(reranked.keys())}

    boosted_docs: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        delta = original_rank[doc_id] - new_rank[doc_id]
        if delta >= 1:
            boosted_docs.append(
                {
                    "doc_id": doc_id,
                    "delta_rank": int(delta),
                    "original_rank": original_rank[doc_id] + 1,
                    "new_rank": new_rank[doc_id] + 1,
                }
            )
    boosted_docs.sort(key=lambda item: (-item["delta_rank"], str(item["doc_id"])))

    explanation_parts = [
        f"Re-ranked {len(doc_ids)} documents with alpha={alpha}.",
        f"Profile terms used: {', '.join(active_terms.keys()) or 'none'}.",
    ]
    if missing_docs:
        explanation_parts.append(
            f"{len(missing_docs)} document(s) not found in MySQL; zero profile boost applied."
        )

    meta = {
        "personalization_applied": True,
        "alpha": alpha,
        "profile_terms_used": list(active_terms.keys()),
        "boosted_docs": boosted_docs,
        "explanation": " ".join(explanation_parts),
        "missing_doc_count": len(missing_docs),
    }
    return reranked, meta
