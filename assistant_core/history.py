import sqlite3
from datetime import datetime
from difflib import SequenceMatcher
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
    """
    CREATE TABLE IF NOT EXISTS response_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        history_id INTEGER NOT NULL,
        username TEXT,
        rating TEXT NOT NULL,
        correction TEXT,
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(history_id),
        FOREIGN KEY(history_id) REFERENCES conversation_history(id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_response_feedback_username
    ON response_feedback(username, updated_at DESC)
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


def save_history(question: str, answer: str, mode: str, sources: list[str], used_web: bool, username: str | None = None) -> int:
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
    history_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return history_id


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


def save_feedback(
    history_id: int,
    rating: str,
    username: str | None = None,
    correction: str | None = None,
    notes: str | None = None,
) -> int:
    ensure_history_schema()
    now = datetime.now().isoformat()
    conn = sqlite3.connect(WAREHOUSE_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO response_feedback(history_id, username, rating, correction, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(history_id) DO UPDATE SET
            username = excluded.username,
            rating = excluded.rating,
            correction = excluded.correction,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (history_id, username, rating, correction, notes, now, now),
    )
    feedback_id = int(cursor.lastrowid or 0)
    conn.commit()
    if not feedback_id:
        row = cursor.execute("SELECT id FROM response_feedback WHERE history_id = ?", (history_id,)).fetchone()
        feedback_id = int(row[0]) if row else 0
    conn.close()
    return feedback_id


def fetch_feedback(limit: int = 30, username: str | None = None) -> list[dict[str, Any]]:
    ensure_history_schema()
    conn = sqlite3.connect(WAREHOUSE_DB)
    conn.row_factory = sqlite3.Row
    base_query = """
        SELECT
            rf.id,
            rf.history_id,
            rf.username,
            rf.rating,
            rf.correction,
            rf.notes,
            rf.created_at,
            rf.updated_at,
            ch.question,
            ch.answer,
            ch.mode,
            ch.sources
        FROM response_feedback rf
        JOIN conversation_history ch ON ch.id = rf.history_id
    """
    if username:
        rows = conn.execute(
            base_query + " WHERE rf.username = ? ORDER BY rf.updated_at DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    else:
        rows = conn.execute(base_query + " ORDER BY rf.updated_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_feedback_memory(question: str, username: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
    ensure_history_schema()
    conn = sqlite3.connect(WAREHOUSE_DB)
    conn.row_factory = sqlite3.Row
    params: tuple[Any, ...]
    if username:
        rows = conn.execute(
            """
            SELECT
                rf.id,
                rf.history_id,
                rf.username,
                rf.rating,
                rf.correction,
                rf.notes,
                rf.updated_at,
                ch.question,
                ch.answer
            FROM response_feedback rf
            JOIN conversation_history ch ON ch.id = rf.history_id
            WHERE rf.rating = 'bad'
              AND rf.username = ?
              AND rf.correction IS NOT NULL
              AND trim(rf.correction) <> ''
            ORDER BY rf.updated_at DESC
            LIMIT 80
            """,
            (username,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
                rf.id,
                rf.history_id,
                rf.username,
                rf.rating,
                rf.correction,
                rf.notes,
                rf.updated_at,
                ch.question,
                ch.answer
            FROM response_feedback rf
            JOIN conversation_history ch ON ch.id = rf.history_id
            WHERE rf.rating = 'bad'
              AND rf.correction IS NOT NULL
              AND trim(rf.correction) <> ''
            ORDER BY rf.updated_at DESC
            LIMIT 120
            """
        ).fetchall()
    conn.close()

    normalized_question = _normalize_feedback_text(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        item = dict(row)
        prior_question = _normalize_feedback_text(item.get("question") or "")
        if not prior_question:
            continue
        ratio = SequenceMatcher(None, normalized_question, prior_question).ratio()
        overlap = len(set(normalized_question.split()) & set(prior_question.split()))
        score = ratio + overlap * 0.08
        if ratio >= 0.58 or overlap >= 3:
            item["similarity"] = round(score, 4)
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("updated_at") or ""))
    return [item for _, item in scored[:limit]]


def _normalize_feedback_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())
