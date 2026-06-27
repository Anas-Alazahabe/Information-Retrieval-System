"""Batch-fetch document text from MySQL (shared by personalization and UI)."""

from typing import Dict, List

from shared.db_config import get_connection


def fetch_document_texts(doc_ids: List[str]) -> Dict[str, str]:
    """Return doc_id -> content for IDs found in the documents table."""
    if not doc_ids:
        return {}

    unique_ids = list(dict.fromkeys(doc_ids))
    placeholders = ", ".join(["%s"] * len(unique_ids))
    sql = f"SELECT id, content FROM documents WHERE id IN ({placeholders})"

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(unique_ids))
            rows = cursor.fetchall()
            cursor.close()
        return {str(doc_id): str(content) for doc_id, content in rows}
    except Exception:
        return {}
