"""One-shot setup for personalization MySQL (schema + optional document migration)."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

import mysql.connector

from scripts.init_personalization_schema import init_schema
from shared.db_config import get_db_config, get_connection


def ensure_database() -> None:
    config = get_db_config()
    db_name = config["database"]
    bootstrap = {k: v for k, v in config.items() if k != "database"}
    conn = mysql.connector.connect(**bootstrap)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {db_name} "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()
    print(f"Database `{db_name}` ready.")


def report_status() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        cursor.close()
    print(f"documents: {doc_count:,}")
    print(f"users: {user_count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup personalization MySQL")
    parser.add_argument(
        "--migrate-docs",
        action="store_true",
        help="Run MS MARCO document migration (slow, ~200K rows)",
    )
    parser.add_argument("--max-docs", type=int, default=200_000)
    parser.add_argument("--skip-schema", action="store_true")
    args = parser.parse_args()

    ensure_database()
    if not args.skip_schema:
        init_schema()

    if args.migrate_docs:
        from migrate_to_db import migrate_data

        migrate_data(max_docs=args.max_docs)

    report_status()


if __name__ == "__main__":
    main()
