"""MySQL connection settings shared across IR services."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import mysql.connector
from mysql.connector import MySQLConnection

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_db_config() -> Dict[str, Any]:
    """Build MySQL connection kwargs from environment variables."""
    return {
        "host": os.environ.get("MYSQL_HOST", "localhost"),
        "user": os.environ.get("MYSQL_USER", "root"),
        "password": os.environ.get("MYSQL_PASSWORD", "MySecretPassword"),
        "database": os.environ.get("MYSQL_DATABASE", "ir_system"),
        "charset": "utf8mb4",
    }


@contextmanager
def get_connection(database: Optional[str] = None) -> Iterator[MySQLConnection]:
    """Context manager yielding a MySQL connection (commits on success)."""
    config = get_db_config()
    if database is not None:
        config = {**config, "database": database}
    conn = mysql.connector.connect(**config)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_db_connection() -> Dict[str, Any]:
    """Return connectivity status for health checks."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
        return {"connected": True, "error": None}
    except Exception as exc:
        return {"connected": False, "error": str(exc)}
