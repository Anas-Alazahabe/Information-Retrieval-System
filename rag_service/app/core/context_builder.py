"""Build RAG context blocks from ranked retrieval results."""

from typing import Any, Dict, List, Tuple

from shared.doc_store import fetch_document_texts

from .token_budget import apply_char_budget


def _sorted_doc_ids(results: Dict[str, float], top_n: int) -> List[Tuple[str, float]]:
    ranked = sorted(results.items(), key=lambda item: item[1], reverse=True)
    return [(str(doc_id), float(score)) for doc_id, score in ranked[:top_n]]


def build_context(
    results: Dict[str, float],
    *,
    top_context_docs: int,
    max_context_chars: int,
) -> Dict[str, Any]:
    """Fetch passage texts and format context for the LLM."""
    ranked = _sorted_doc_ids(results, top_context_docs)
    doc_ids = [doc_id for doc_id, _ in ranked]
    texts = fetch_document_texts(doc_ids)

    missing_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in texts]
    found: List[Tuple[str, float, str]] = [
        (doc_id, score, texts[doc_id]) for doc_id, score in ranked if doc_id in texts
    ]

    kept, total_chars = apply_char_budget(found, max_context_chars=max_context_chars)
    context_doc_ids = [doc_id for doc_id, _, _ in kept]

    blocks: List[str] = []
    citations: List[Dict[str, Any]] = []
    for doc_id, score, text in kept:
        blocks.append(f"[DOC {doc_id} score={score:.2f}]\n{text}")
        citations.append(
            {
                "doc_id": doc_id,
                "snippet": text[:500] + ("…" if len(text) > 500 else ""),
                "retrieval_score": round(score, 4),
            }
        )

    return {
        "context_text": "\n\n".join(blocks),
        "context_doc_ids": context_doc_ids,
        "missing_doc_ids": missing_doc_ids,
        "citations": citations,
        "context_chars": total_chars,
    }
