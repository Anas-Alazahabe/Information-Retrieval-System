"""Gemini REST API client for RAG answer generation."""

from typing import Any, Dict, Optional

import requests

from app.core.prompts import SYSTEM_INSTRUCTION, build_user_prompt

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(Exception):
    """Raised when the Gemini API returns an error."""


def generate_answer(
    query: str,
    context_text: str,
    *,
    api_key: str,
    model: str,
    timeout: int = 60,
) -> str:
    """Call Gemini generateContent and return the answer text."""
    if not api_key.strip():
        raise GeminiError("GEMINI_API_KEY is not configured.")

    url = f"{GEMINI_API_BASE}/{model}:generateContent"
    user_prompt = build_user_prompt(query, context_text)
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    try:
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": api_key,
            },
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise GeminiError(f"Gemini request failed: {exc}") from exc

    if response.status_code != 200:
        detail = response.text[:500]
        raise GeminiError(f"Gemini API error {response.status_code}: {detail}")

    data = response.json()
    return _extract_text(data)


def _extract_text(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        texts = [part.get("text", "") for part in parts if part.get("text")]
        if texts:
            return "\n".join(texts).strip()

    block_reason = (candidates[0].get("finishReason") if candidates else None) or "unknown"
    raise GeminiError(f"Gemini returned no text (finishReason={block_reason})")
