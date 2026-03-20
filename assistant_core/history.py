import sqlite3
from datetime import datetime
from typing import Any

from assistant_core.config import WAREHOUSE_DB


HISTORY_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS conversation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        mode TEXT,
        sources TEXT,
        used_web INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversation_history_created_at
    ON conversation_history(created_at DESC)
    """,
]


def ensure_history_schema() -> None:
    conn = sqlite3.connect(WAREHOUSE_DB)
    cursor = conn.cursor()
    for statement in HISTORY_SCHEMA:
        cursor.execute(statement)
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(conversation_history)").fetchall()}
    if "username" not in columns:
        cursor.execute("ALTER TABLE conversation_history ADD COLUMN username TEXT")
    conn.commit()
    conn.close()


def save_history(question: str, answer: str, mode: str, sources: list[str], used_web: bool, username: str | None = None) -> None:
    ensure_history_schema()
    conn = sqlite3.connect(WAREHOUSE_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO conversation_history(username, question, answer, mode, sources, used_web, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            question,
            answer,
            mode,
            ", ".join(sources),
            1 if used_web else 0,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def fetch_history(limit: int = 20, username: str | None = None) -> list[dict[str, Any]]:
    ensure_history_schema()
    conn = sqlite3.connect(WAREHOUSE_DB)
    conn.row_factory = sqlite3.Row
    if username:
        rows = conn.execute(
            """
            SELECT id, username, question, answer, mode, sources, used_web, created_at
            FROM conversation_history
            WHERE username = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (username, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, username, question, answer, mode, sources, used_web, created_at
            FROM conversation_history
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
