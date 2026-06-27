"""Tests for RAG context assembly and char budget."""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_service.app.core.context_builder import build_context
from rag_service.app.core.token_budget import apply_char_budget


def test_apply_char_budget_drops_lowest_scored_first():
    docs = [
        ("a", 10.0, "A" * 100),
        ("b", 5.0, "B" * 100),
        ("c", 1.0, "C" * 100),
    ]
    kept, total = apply_char_budget(docs, max_context_chars=250)
    assert [d[0] for d in kept] == ["a", "b"]
    assert total == 200


def test_build_context_orders_by_score_and_formats():
    results = {"low": 1.0, "high": 9.0, "mid": 5.0}
    with patch(
        "rag_service.app.core.context_builder.fetch_document_texts",
        return_value={
            "high": "High passage text.",
            "mid": "Mid passage text.",
            "low": "Low passage text.",
        },
    ):
        ctx = build_context(results, top_context_docs=2, max_context_chars=12000)

    assert ctx["context_doc_ids"] == ["high", "mid"]
    assert "High passage text." in ctx["context_text"]
    assert ctx["missing_doc_ids"] == []
    assert len(ctx["citations"]) == 2


def test_build_context_tracks_missing_docs():
    results = {"found": 5.0, "missing": 4.0}
    with patch(
        "rag_service.app.core.context_builder.fetch_document_texts",
        return_value={"found": "Found text."},
    ):
        ctx = build_context(results, top_context_docs=2, max_context_chars=12000)

    assert ctx["context_doc_ids"] == ["found"]
    assert ctx["missing_doc_ids"] == ["missing"]
