import sqlite3
from pathlib import Path

import fitz

from assistant_core.config import DOC_DIR, WAREHOUSE_DB


DOC_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT UNIQUE,
        file_path TEXT,
        content TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        chunk_index INTEGER,
        content TEXT
    )
    """,
]


def _chunk_text(text: str, chunk_size: int = 1200) -> list[str]:
    return [text[index:index + chunk_size] for index in range(0, len(text), chunk_size) if text[index:index + chunk_size].strip()]


def _extract_pdf_text(path: Path) -> str:
    pdf = fitz.open(path)
    try:
        pages = [page.get_text("text") for page in pdf]
    finally:
        pdf.close()
    return "\n".join(page.strip() for page in pages if page and page.strip())


def index_documents(folder: Path | None = None) -> int:
    target = folder or DOC_DIR
    target.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(WAREHOUSE_DB)
    cursor = conn.cursor()

    for statement in DOC_SCHEMA:
        cursor.execute(statement)

    cursor.execute("DELETE FROM document_chunks")
    cursor.execute("DELETE FROM documents")

    total_documents = 0

    for pdf_path in sorted(target.glob("*.pdf")):
        content = _extract_pdf_text(pdf_path)
        cursor.execute(
            """
            INSERT INTO documents(file_name, file_path, content)
            VALUES (?, ?, ?)
            """,
            (pdf_path.name, str(pdf_path), content),
        )
        document_id = cursor.lastrowid
        for index, chunk in enumerate(_chunk_text(content)):
            cursor.execute(
                """
                INSERT INTO document_chunks(document_id, chunk_index, content)
                VALUES (?, ?, ?)
                """,
                (document_id, index, chunk),
            )
        total_documents += 1

    conn.commit()
    conn.close()
    return total_documents


def indexed_documents_count() -> int:
    conn = sqlite3.connect(WAREHOUSE_DB)
    try:
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()
