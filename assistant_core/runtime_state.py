import sqlite3
from datetime import datetime
from typing import Any

from assistant_core.config import WAREHOUSE_DB


RUNTIME_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS runtime_state (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT NOT NULL
    )
    """
]


def ensure_runtime_schema() -> None:
    conn = sqlite3.connect(WAREHOUSE_DB)
    cursor = conn.cursor()
    for statement in RUNTIME_SCHEMA:
        cursor.execute(statement)
    conn.commit()
    conn.close()


def set_runtime_value(key: str, value: str) -> None:
    ensure_runtime_schema()
    conn = sqlite3.connect(WAREHOUSE_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO runtime_state(key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """,
        (key, value, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_runtime_value(key: str) -> dict[str, Any] | None:
    ensure_runtime_schema()
    conn = sqlite3.connect(WAREHOUSE_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT key, value, updated_at FROM runtime_state WHERE key = ?",
        (key,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None

