"""Tests for search_with_rag orchestration."""

from unittest.mock import MagicMock, patch

from shared.search_pipeline import search_with_rag


def _mock_personalization_pipeline():
  return {
      "search": {
          "status": "success",
          "results": {"doc1": 8.5, "doc2": 6.0},
          "total_results": 2,
      },
      "refinement": None,
      "personalization": None,
  }


def test_search_with_rag_skips_when_disabled():
    with patch(
        "shared.search_pipeline.search_with_personalization",
        return_value=_mock_personalization_pipeline(),
    ) as mock_search:
        result = search_with_rag(
            raw_query="how to tie a tie",
            representation_mode="bm25",
            use_refinement=False,
            use_personalization=False,
            use_rag=False,
            techniques=[],
        )

    mock_search.assert_called_once()
    assert result["rag"] is None


def test_search_with_rag_calls_generate_when_enabled():
    rag_response = MagicMock()
    rag_response.json.return_value = {
        "status": "success",
        "answer": "Tie instructions.",
        "context_doc_ids": ["doc1"],
    }
    rag_response.raise_for_status = MagicMock()

    with patch(
        "shared.search_pipeline.search_with_personalization",
        return_value=_mock_personalization_pipeline(),
    ), patch("shared.search_pipeline.requests.post", return_value=rag_response) as mock_post:
        result = search_with_rag(
            raw_query="how to tie a tie",
            representation_mode="bm25",
            use_refinement=False,
            use_personalization=False,
            use_rag=True,
            techniques=[],
            rag_top_context_docs=3,
        )

    assert mock_post.called
    assert result["rag"]["answer"] == "Tie instructions."
