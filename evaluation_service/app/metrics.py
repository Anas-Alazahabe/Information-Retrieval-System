import math
from typing import Dict, List, Sequence, Set


def _binary_relevant(qrels: Dict[str, int]) -> Set[str]:
    """تحويل qrels إلى مجموعة وثائق ذات صلة (relevance > 0)."""
    return {doc_id for doc_id, grade in qrels.items() if grade > 0}


def precision_at_k(ranked_docs: Sequence[str], relevant: Set[str], k: int) -> float:
    """حساب Precision@k."""
    if k <= 0:
        return 0.0
    top_k = ranked_docs[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(ranked_docs: Sequence[str], relevant: Set[str], k: int) -> float:
    """حساب Recall@k."""
    if not relevant:
        return 0.0
    top_k = ranked_docs[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def average_precision(ranked_docs: Sequence[str], relevant: Set[str]) -> float:
    """حساب Average Precision لاستعلام واحد."""
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def dcg_at_k(ranked_docs: Sequence[str], qrels: Dict[str, int], k: int) -> float:
    """حساب DCG@k بناءً على درجات الصلة في qrels."""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_docs[:k], start=1):
        rel = qrels.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_docs: Sequence[str], qrels: Dict[str, int], k: int) -> float:
    """حساب nDCG@k عبر تطبيع DCG على الترتيب المثالي."""
    ideal_docs = sorted(qrels.items(), key=lambda item: item[1], reverse=True)
    ideal_ranked = [doc_id for doc_id, grade in ideal_docs if grade > 0]
    ideal_dcg = dcg_at_k(ideal_ranked, qrels, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_docs, qrels, k) / ideal_dcg


def aggregate_metrics(
  per_query_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """تجميع مقاييس جميع الاستعلامات إلى تقرير نهائي لكل نمط."""
    if not per_query_metrics:
        return {
            "map": 0.0,
            "recall": 0.0,
            "precision_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "num_queries": 0,
        }
    n = len(per_query_metrics)
    return {
        "map": round(sum(m["ap"] for m in per_query_metrics) / n, 6),
        "recall": round(sum(m["recall"] for m in per_query_metrics) / n, 6),
        "precision_at_10": round(sum(m["precision_at_10"] for m in per_query_metrics) / n, 6),
        "ndcg_at_10": round(sum(m["ndcg_at_10"] for m in per_query_metrics) / n, 6),
        "num_queries": n,
    }


def evaluate_ranked_list(
    ranked_docs: Sequence[str],
    qrels: Dict[str, int],
    k: int = 10,
) -> Dict[str, float]:
    """إرجاع المقاييس الأساسية لاستعلام واحد."""
    relevant = _binary_relevant(qrels)
    return {
        "ap": round(average_precision(ranked_docs, relevant), 6),
        "recall": round(recall_at_k(ranked_docs, relevant, k), 6),
        "precision_at_10": round(precision_at_k(ranked_docs, relevant, k), 6),
        "ndcg_at_10": round(ndcg_at_k(ranked_docs, qrels, k), 6),
    }
