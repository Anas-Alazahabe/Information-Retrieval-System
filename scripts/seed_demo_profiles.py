"""Seed demo user profiles for personalization UI presentation.

Demo queries:
  demo_health: diabetes symptoms, heart disease treatment, hospital care
  demo_tech: python programming tutorial, machine learning basics, database design

Run after MySQL schema init and personalization service is reachable:
  python scripts/seed_demo_profiles.py
"""

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.db_config import get_connection
from shared.ir_config import PERSONALIZATION_URL

DEMO_PROFILES = {
    "demo_health": {
        "queries": [
            "what are diabetes symptoms",
            "heart disease treatment options",
            "hospital patient care guidelines",
            "blood pressure medication side effects",
            "cancer screening recommendations",
        ],
        "click_terms": ["diabetes", "hospital", "cancer"],
    },
    "demo_tech": {
        "queries": [
            "how to learn python programming",
            "machine learning tutorial for beginners",
            "database design best practices",
            "javascript web development guide",
            "linux command line basics",
        ],
        "click_terms": ["python", "programming", "database"],
    },
}


def _find_doc_for_term(term: str) -> str | None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM documents WHERE LOWER(content) LIKE %s LIMIT 1",
            (f"%{term.lower()}%",),
        )
        row = cursor.fetchone()
        cursor.close()
    return str(row[0]) if row else None


def _reset_user(user_id: str) -> None:
    requests.delete(
        f"{PERSONALIZATION_URL.rstrip('/')}/profile/{user_id}",
        timeout=10,
    )


def _log_query(user_id: str, query_text: str) -> None:
    requests.post(
        f"{PERSONALIZATION_URL.rstrip('/')}/events/query",
        json={"user_id": user_id, "query_text": query_text},
        timeout=10,
    ).raise_for_status()


def _log_click(user_id: str, doc_id: str, query_text: str) -> None:
    requests.post(
        f"{PERSONALIZATION_URL.rstrip('/')}/events/click",
        json={"user_id": user_id, "doc_id": doc_id, "query_text": query_text},
        timeout=10,
    ).raise_for_status()


def seed_profiles(reset: bool = True) -> None:
    health = requests.get(f"{PERSONALIZATION_URL.rstrip('/')}/health", timeout=5)
    health.raise_for_status()
    if not health.json().get("database_connected"):
        raise RuntimeError(
            "Personalization service is up but MySQL is not connected. "
            "Start MySQL, run migrate_to_db.py and init_personalization_schema.py."
        )

    for user_id, profile in DEMO_PROFILES.items():
        if reset:
            _reset_user(user_id)
        for query_text in profile["queries"]:
            _log_query(user_id, query_text)
        for term in profile.get("click_terms", []):
            doc_id = _find_doc_for_term(term)
            if doc_id:
                _log_click(user_id, doc_id, profile["queries"][0])

        response = requests.get(
            f"{PERSONALIZATION_URL.rstrip('/')}/profile/{user_id}",
            timeout=10,
        )
        response.raise_for_status()
        summary = response.json()
        terms = list((summary.get("interest_terms") or {}).keys())[:8]
        print(f"{user_id}: queries={summary.get('query_count')} clicks={summary.get('click_count')} terms={terms}")


if __name__ == "__main__":
    seed_profiles()
