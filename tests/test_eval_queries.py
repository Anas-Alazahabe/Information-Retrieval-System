"""Tests for shared evaluation query selection."""

from evaluation_service.app.eval_queries import build_eval_protocol, select_eval_queries


def test_select_eval_queries_only_judged():
    queries = {"q1": "a", "q2": "b", "q3": "c", "q4": "d"}
    qrels = {"q1": {"d1": 1}, "q3": {"d2": 1}}
    order = ["q1", "q2", "q3", "q4"]
    selected = select_eval_queries(queries, qrels, max_queries=3, query_order=order)
    assert selected == [("q1", "a"), ("q3", "c")]


def test_select_eval_queries_respects_cap():
    queries = {f"q{i}": f"text{i}" for i in range(5)}
    qrels = {f"q{i}": {"d1": 1} for i in range(5)}
    order = [f"q{i}" for i in range(5)]
    selected = select_eval_queries(queries, qrels, max_queries=2, query_order=order)
    assert len(selected) == 2
    assert selected[0][0] == "q0"
    assert selected[1][0] == "q1"


def test_build_eval_protocol():
    protocol = build_eval_protocol(
        dataset_name="msmarco-passage/dev",
        num_judged_queries=59,
        max_queries=100,
    )
    assert protocol["excludes_ui_queries"] is True
    assert protocol["num_judged_queries"] == 59
