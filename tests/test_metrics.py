from evaluation_service.app.metrics import (
    aggregate_metrics,
    evaluate_ranked_list,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_10():
    ranked = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]
    relevant = {"d1", "d3", "d99"}
    assert precision_at_k(ranked, relevant, 10) == 0.2


def test_recall_at_10():
    ranked = ["d1", "d2", "d3"]
    relevant = {"d1", "d3", "d99"}
    assert recall_at_k(ranked, relevant, 10) == 2 / 3


def test_evaluate_ranked_list():
    qrels = {"d1": 1, "d2": 0, "d3": 2}
    metrics = evaluate_ranked_list(["d3", "d1", "d2"], qrels, k=10)
    assert metrics["precision_at_10"] > 0
    assert metrics["recall"] > 0
    assert metrics["ap"] > 0
    assert metrics["ndcg_at_10"] > 0


def test_aggregate_metrics():
    per_query = [
        {"ap": 1.0, "recall": 1.0, "precision_at_10": 0.5, "ndcg_at_10": 0.8},
        {"ap": 0.5, "recall": 0.5, "precision_at_10": 0.3, "ndcg_at_10": 0.4},
    ]
    agg = aggregate_metrics(per_query)
    assert agg["map"] == 0.75
    assert agg["num_queries"] == 2


def test_ndcg_zero_when_no_relevant():
    assert ndcg_at_k(["d1", "d2"], {}, 10) == 0.0
