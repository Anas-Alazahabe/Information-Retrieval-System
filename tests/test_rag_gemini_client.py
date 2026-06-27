"""Tests for Gemini REST client."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_service.app.core.gemini_client import GeminiError, generate_answer


def test_generate_answer_extracts_text():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Generated answer."}],
                }
            }
        ]
    }

    with patch("rag_service.app.core.gemini_client.requests.post", return_value=mock_response):
        text = generate_answer(
            "question?",
            "context block",
            api_key="key",
            model="gemini-flash-latest",
        )

    assert text == "Generated answer."


def test_generate_answer_missing_key_raises():
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        generate_answer("q", "ctx", api_key="", model="gemini-flash-latest")


def test_generate_answer_api_error_raises():
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden"

    with patch("rag_service.app.core.gemini_client.requests.post", return_value=mock_response):
        with pytest.raises(GeminiError, match="403"):
            generate_answer("q", "ctx", api_key="key", model="gemini-flash-latest")
