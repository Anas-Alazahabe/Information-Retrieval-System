"""MySQL persistence for user profiles and events."""

from typing import Dict, Optional, Tuple

from shared.db_config import get_connection
from shared.ir_config import PROFILE_MAX_CLICKS, PROFILE_MAX_QUERIES, PROFILE_TOP_TERMS


def ensure_user(user_id: str, display_name: Optional[str] = None) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (user_id, display_name)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE display_name = COALESCE(VALUES(display_name), display_name)
            """,
            (user_id, display_name or user_id),
        )
        cursor.close()


def log_query_event(user_id: str, query_text: str) -> None:
    ensure_user(user_id)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_query_events (user_id, query_text) VALUES (%s, %s)",
            (user_id, query_text),
        )
        _prune_old_events(cursor, user_id, "user_query_events", PROFILE_MAX_QUERIES)
        cursor.close()


def log_click_event(user_id: str, doc_id: str, query_text: Optional[str] = None) -> None:
    ensure_user(user_id)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_click_events (user_id, doc_id, query_text)
            VALUES (%s, %s, %s)
            """,
            (user_id, doc_id, query_text),
        )
        _prune_old_events(cursor, user_id, "user_click_events", PROFILE_MAX_CLICKS)
        cursor.close()


def _prune_old_events(cursor, user_id: str, table: str, max_rows: int) -> None:
    cursor.execute(
        f"""
        DELETE FROM {table}
        WHERE user_id = %s AND id NOT IN (
            SELECT id FROM (
                SELECT id FROM {table}
                WHERE user_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
            ) AS recent
        )
        """,
        (user_id, user_id, max_rows),
    )


def upsert_interest_terms(
    user_id: str,
    term_weights: Dict[str, float],
    source: str,
) -> None:
    if not term_weights:
        return

    ensure_user(user_id)
    with get_connection() as conn:
        cursor = conn.cursor()
        for term, weight in term_weights.items():
            if weight <= 0:
                continue
            cursor.execute(
                """
                INSERT INTO user_interest_terms (user_id, term, weight, source)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    weight = weight + VALUES(weight),
                    source = VALUES(source),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, term, weight, source),
            )
        _prune_interest_terms(cursor, user_id)
        cursor.close()


def _prune_interest_terms(cursor, user_id: str) -> None:
    cursor.execute(
        """
        DELETE FROM user_interest_terms
        WHERE user_id = %s AND term NOT IN (
            SELECT term FROM (
                SELECT term FROM user_interest_terms
                WHERE user_id = %s
                ORDER BY weight DESC, term ASC
                LIMIT %s
            ) AS top_terms
        )
        """,
        (user_id, user_id, PROFILE_TOP_TERMS),
    )


def get_interest_terms(user_id: str) -> Dict[str, float]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT term, weight FROM user_interest_terms
            WHERE user_id = %s
            ORDER BY weight DESC, term ASC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
    return {term: float(weight) for term, weight in rows}


def get_profile_summary(user_id: str) -> Tuple[Dict[str, float], int, int]:
    interest_terms = get_interest_terms(user_id)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM user_query_events WHERE user_id = %s",
            (user_id,),
        )
        query_count = int(cursor.fetchone()[0])
        cursor.execute(
            "SELECT COUNT(*) FROM user_click_events WHERE user_id = %s",
            (user_id,),
        )
        click_count = int(cursor.fetchone()[0])
        cursor.close()
    return interest_terms, query_count, click_count


def reset_profile(user_id: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        for table in (
            "user_interest_terms",
            "user_query_events",
            "user_click_events",
            "users",
        ):
            cursor.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
        cursor.close()


def get_documents_count() -> Optional[int]:
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = int(cursor.fetchone()[0])
            cursor.close()
        return count
    except Exception:
        return None
