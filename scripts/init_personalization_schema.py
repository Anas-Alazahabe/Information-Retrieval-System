"""Create personalization tables in MySQL (idempotent)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.db_config import get_connection

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(64) PRIMARY KEY,
        display_name VARCHAR(128),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS user_query_events (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(64) NOT NULL,
        query_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_query_time (user_id, created_at)
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS user_click_events (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(64) NOT NULL,
        doc_id VARCHAR(255) NOT NULL,
        query_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_click_time (user_id, created_at)
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS user_interest_terms (
        user_id VARCHAR(64) NOT NULL,
        term VARCHAR(64) NOT NULL,
        weight DOUBLE NOT NULL DEFAULT 0,
        source ENUM('query', 'click') NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, term)
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
]


def init_schema() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        for statement in SCHEMA_STATEMENTS:
            cursor.execute(statement)
        cursor.close()
    print("Personalization schema initialized successfully.")


if __name__ == "__main__":
    init_schema()
