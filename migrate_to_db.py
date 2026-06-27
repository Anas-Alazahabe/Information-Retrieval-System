import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

import ir_datasets

from shared.db_config import get_connection
from shared.ir_datasets_patch import patch_ir_datasets_tsv_utf8


def migrate_data(max_docs: int = 200_000) -> None:
    try:
        patch_ir_datasets_tsv_utf8()
        with get_connection() as conn:
            cursor = conn.cursor()

            print("Creating table...")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(255) PRIMARY KEY,
                    content TEXT NOT NULL
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
                """
            )

            print("Loading dataset...")
            dataset = ir_datasets.load("msmarco-passage")

            batch_size = 1000
            batch = []
            count = 0

            print("Starting data insertion...")
            sql = "INSERT IGNORE INTO documents (id, content) VALUES (%s, %s)"

            for doc in dataset.docs_iter():
                batch.append((doc.doc_id, doc.text))
                count += 1

                if len(batch) == batch_size:
                    try:
                        cursor.executemany(sql, batch)
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        for item in batch:
                            try:
                                cursor.execute(sql, item)
                                conn.commit()
                            except Exception:
                                conn.rollback()

                    print(f"Processed {count} records...")
                    batch = []

                if count >= max_docs:
                    break

            if batch:
                try:
                    cursor.executemany(sql, batch)
                    conn.commit()
                except Exception:
                    pass

            cursor.execute("SELECT COUNT(*) FROM documents")
            total = cursor.fetchone()[0]
            print(f"Done. documents table row count: {total}")
            cursor.close()

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate MS MARCO passages to MySQL")
    parser.add_argument("--max-docs", type=int, default=200_000)
    args = parser.parse_args()
    migrate_data(max_docs=args.max_docs)
