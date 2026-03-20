import sqlite3
from dataclasses import dataclass
from typing import Any

from assistant_core.config import RAW_MODULE_FILES, WAREHOUSE_DB
from assistant_core.utils import (
    compact_text,
    entity_fields,
    load_json,
    nested_value,
    owner_fields,
    parse_datetime,
    strip_html,
)


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS owners (
        id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS leads (
        id TEXT PRIMARY KEY,
        owner_id TEXT,
        owner_name TEXT,
        owner_email TEXT,
        company_name TEXT,
        contact_name TEXT,
        full_name TEXT,
        email TEXT,
        phone TEXT,
        city TEXT,
        state TEXT,
        address TEXT,
        website TEXT,
        giro TEXT,
        otros_datos TEXT,
        unit_count INTEGER,
        unit_type TEXT,
        phase TEXT,
        last_activity_time TEXT,
        created_time TEXT,
        modified_time TEXT,
        raw_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS contacts (
        id TEXT PRIMARY KEY,
        owner_id TEXT,
        owner_name TEXT,
        owner_email TEXT,
        company_name TEXT,
        contact_name TEXT,
        full_name TEXT,
        email TEXT,
        phone TEXT,
        city TEXT,
        state TEXT,
        address TEXT,
        website TEXT,
        giro TEXT,
        otros_datos TEXT,
        unit_count INTEGER,
        unit_type TEXT,
        phase TEXT,
        last_activity_time TEXT,
        created_time TEXT,
        modified_time TEXT,
        raw_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notes (
        id TEXT PRIMARY KEY,
        owner_id TEXT,
        owner_name TEXT,
        owner_email TEXT,
        parent_id TEXT,
        parent_name TEXT,
        parent_module TEXT,
        title TEXT,
        content_raw TEXT,
        content_text TEXT,
        created_time TEXT,
        modified_time TEXT,
        raw_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calls (
        id TEXT PRIMARY KEY,
        owner_id TEXT,
        owner_name TEXT,
        owner_email TEXT,
        contact_id TEXT,
        contact_name TEXT,
        contact_module TEXT,
        subject TEXT,
        status TEXT,
        call_type TEXT,
        call_result TEXT,
        start_time TEXT,
        created_time TEXT,
        modified_time TEXT,
        description TEXT,
        raw_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        owner_id TEXT,
        owner_name TEXT,
        owner_email TEXT,
        contact_id TEXT,
        contact_name TEXT,
        contact_module TEXT,
        subject TEXT,
        status TEXT,
        priority TEXT,
        due_date TEXT,
        closed_time TEXT,
        created_time TEXT,
        modified_time TEXT,
        description TEXT,
        raw_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        owner_id TEXT,
        owner_name TEXT,
        owner_email TEXT,
        contact_id TEXT,
        contact_name TEXT,
        contact_module TEXT,
        title TEXT,
        start_time TEXT,
        end_time TEXT,
        created_time TEXT,
        modified_time TEXT,
        description TEXT,
        raw_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS interactions (
        source_type TEXT,
        source_id TEXT PRIMARY KEY,
        owner_id TEXT,
        owner_name TEXT,
        owner_email TEXT,
        related_id TEXT,
        related_name TEXT,
        related_module TEXT,
        interaction_at TEXT,
        status TEXT,
        summary TEXT,
        detail TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_interactions_owner_time
    ON interactions(owner_id, interaction_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_interactions_related_time
    ON interactions(related_id, interaction_at)
    """,
]


@dataclass
class BuildStats:
    leads: int = 0
    contacts: int = 0
    notes: int = 0
    calls: int = 0
    tasks: int = 0
    events: int = 0
    interactions: int = 0


def _json_text(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def _normalize_states(value: Any) -> str | None:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return value or None


def _extract_person_record(payload: dict[str, Any]) -> dict[str, Any]:
    owner_id, owner_name, owner_email = owner_fields(payload)
    return {
        "id": payload.get("id"),
        "owner_id": owner_id,
        "owner_name": owner_name,
        "owner_email": owner_email,
        "company_name": nested_value(payload, "Empresa"),
        "contact_name": nested_value(payload, "Nombre_contacto"),
        "full_name": nested_value(payload, "Full_Name", "Last_Name"),
        "email": nested_value(payload, "Email"),
        "phone": nested_value(payload, "Phone", "Mobile"),
        "city": nested_value(payload, "Ciudad"),
        "state": _normalize_states(payload.get("Estado")),
        "address": nested_value(payload, "Direcci_n"),
        "website": nested_value(payload, "Sitio_Web"),
        "giro": nested_value(payload, "Giro"),
        "otros_datos": nested_value(payload, "Otros_datos"),
        "unit_count": nested_value(payload, "N_mero_Unidades"),
        "unit_type": nested_value(payload, "Tipo_de_unidades"),
        "phase": nested_value(payload, "Fase"),
        "last_activity_time": parse_datetime(payload.get("Last_Activity_Time")),
        "created_time": parse_datetime(payload.get("Created_Time")),
        "modified_time": parse_datetime(payload.get("Modified_Time")),
        "raw_json": _json_text(payload),
    }


def _upsert_owner(cursor: sqlite3.Cursor, owner_id: str | None, owner_name: str | None, owner_email: str | None) -> None:
    if not owner_id:
        return
    cursor.execute(
        """
        INSERT INTO owners(id, name, email)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            email=excluded.email
        """,
        (owner_id, owner_name, owner_email),
    )


def _replace_people_table(cursor: sqlite3.Cursor, table: str, rows: list[dict[str, Any]]) -> int:
    cursor.execute(f"DELETE FROM {table}")
    for row in rows:
        _upsert_owner(cursor, row["owner_id"], row["owner_name"], row["owner_email"])
        cursor.execute(
            f"""
            INSERT INTO {table}(
                id, owner_id, owner_name, owner_email, company_name, contact_name, full_name,
                email, phone, city, state, address, website, giro, otros_datos,
                unit_count, unit_type, phase, last_activity_time, created_time,
                modified_time, raw_json
            ) VALUES (
                :id, :owner_id, :owner_name, :owner_email, :company_name, :contact_name, :full_name,
                :email, :phone, :city, :state, :address, :website, :giro, :otros_datos,
                :unit_count, :unit_type, :phase, :last_activity_time, :created_time,
                :modified_time, :raw_json
            )
            """,
            row,
        )
    return len(rows)


def _load_notes(cursor: sqlite3.Cursor, payloads: list[dict[str, Any]]) -> tuple[int, list[tuple[Any, ...]]]:
    cursor.execute("DELETE FROM notes")
    rows: list[tuple[Any, ...]] = []

    for payload in payloads:
        owner_id, owner_name, owner_email = owner_fields(payload)
        parent_id, parent_name = entity_fields(payload, "Parent_Id")
        parent_module = payload.get("$se_module")
        created_time = parse_datetime(payload.get("Created_Time"))
        modified_time = parse_datetime(payload.get("Modified_Time"))
        content_raw = payload.get("Note_Content") or ""
        content_text = strip_html(content_raw)

        _upsert_owner(cursor, owner_id, owner_name, owner_email)
        row = (
            payload.get("id"),
            owner_id,
            owner_name,
            owner_email,
            parent_id,
            parent_name,
            parent_module,
            payload.get("Note_Title"),
            content_raw,
            content_text,
            created_time,
            modified_time,
            _json_text(payload),
        )
        rows.append(row)

    cursor.executemany(
        """
        INSERT INTO notes(
            id, owner_id, owner_name, owner_email, parent_id, parent_name, parent_module,
            title, content_raw, content_text, created_time, modified_time, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    interactions = [
        (
            "note",
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[10] or row[11],
            None,
            compact_text(row[7], row[9][:180]),
            row[9],
        )
        for row in rows
    ]
    return len(rows), interactions


def _load_calls(cursor: sqlite3.Cursor, payloads: list[dict[str, Any]]) -> tuple[int, list[tuple[Any, ...]]]:
    cursor.execute("DELETE FROM calls")
    rows: list[tuple[Any, ...]] = []

    for payload in payloads:
        owner_id, owner_name, owner_email = owner_fields(payload)
        contact_id, contact_name = entity_fields(payload, "Who_Id")
        module = payload.get("$se_module")
        _upsert_owner(cursor, owner_id, owner_name, owner_email)
        description = strip_html(payload.get("Description"))

        row = (
            payload.get("id"),
            owner_id,
            owner_name,
            owner_email,
            contact_id,
            contact_name,
            module,
            payload.get("Subject"),
            payload.get("Call_Status"),
            payload.get("Call_Type"),
            payload.get("Call_Result"),
            parse_datetime(payload.get("Call_Start_Time")),
            parse_datetime(payload.get("Created_Time")),
            parse_datetime(payload.get("Modified_Time")),
            description,
            _json_text(payload),
        )
        rows.append(row)

    cursor.executemany(
        """
        INSERT INTO calls(
            id, owner_id, owner_name, owner_email, contact_id, contact_name, contact_module,
            subject, status, call_type, call_result, start_time, created_time,
            modified_time, description, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    interactions = [
        (
            "call",
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[11] or row[12] or row[13],
            row[8],
            compact_text(row[7], row[10], row[14][:180]),
            row[14],
        )
        for row in rows
    ]
    return len(rows), interactions


def _load_tasks(cursor: sqlite3.Cursor, payloads: list[dict[str, Any]]) -> tuple[int, list[tuple[Any, ...]]]:
    cursor.execute("DELETE FROM tasks")
    rows: list[tuple[Any, ...]] = []

    for payload in payloads:
        owner_id, owner_name, owner_email = owner_fields(payload)
        contact_id, contact_name = entity_fields(payload, "Who_Id")
        module = payload.get("$se_module")
        _upsert_owner(cursor, owner_id, owner_name, owner_email)
        description = strip_html(payload.get("Description"))

        row = (
            payload.get("id"),
            owner_id,
            owner_name,
            owner_email,
            contact_id,
            contact_name,
            module,
            payload.get("Subject"),
            payload.get("Status"),
            payload.get("Priority"),
            payload.get("Due_Date"),
            parse_datetime(payload.get("Closed_Time")),
            parse_datetime(payload.get("Created_Time")),
            parse_datetime(payload.get("Modified_Time")),
            description,
            _json_text(payload),
        )
        rows.append(row)

    cursor.executemany(
        """
        INSERT INTO tasks(
            id, owner_id, owner_name, owner_email, contact_id, contact_name, contact_module,
            subject, status, priority, due_date, closed_time, created_time,
            modified_time, description, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    interactions = [
        (
            "task",
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[10] or row[12] or row[13],
            row[8],
            compact_text(row[7], row[9], row[14][:180]),
            row[14],
        )
        for row in rows
    ]
    return len(rows), interactions


def _load_events(cursor: sqlite3.Cursor, payloads: list[dict[str, Any]]) -> tuple[int, list[tuple[Any, ...]]]:
    cursor.execute("DELETE FROM events")
    rows: list[tuple[Any, ...]] = []

    for payload in payloads:
        owner_id, owner_name, owner_email = owner_fields(payload)
        contact_id, contact_name = entity_fields(payload, "Who_Id")
        module = payload.get("$se_module")
        _upsert_owner(cursor, owner_id, owner_name, owner_email)
        description = strip_html(payload.get("Description"))

        row = (
            payload.get("id"),
            owner_id,
            owner_name,
            owner_email,
            contact_id,
            contact_name,
            module,
            payload.get("Event_Title"),
            parse_datetime(payload.get("Start_DateTime")),
            parse_datetime(payload.get("End_DateTime")),
            parse_datetime(payload.get("Created_Time")),
            parse_datetime(payload.get("Modified_Time")),
            description,
            _json_text(payload),
        )
        rows.append(row)

    cursor.executemany(
        """
        INSERT INTO events(
            id, owner_id, owner_name, owner_email, contact_id, contact_name, contact_module,
            title, start_time, end_time, created_time, modified_time, description, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    interactions = [
        (
            "event",
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[8] or row[10] or row[11],
            None,
            compact_text(row[7], row[12][:180]),
            row[12],
        )
        for row in rows
    ]
    return len(rows), interactions


def _rebuild_interactions(cursor: sqlite3.Cursor, rows: list[tuple[Any, ...]]) -> int:
    cursor.execute("DELETE FROM interactions")
    cursor.executemany(
        """
        INSERT INTO interactions(
            source_type, source_id, owner_id, owner_name, owner_email,
            related_id, related_name, related_module, interaction_at,
            status, summary, detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def build_warehouse() -> BuildStats:
    WAREHOUSE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(WAREHOUSE_DB)
    cursor = conn.cursor()

    for statement in SCHEMA_STATEMENTS:
        cursor.execute(statement)

    leads = [_extract_person_record(item) for item in load_json(RAW_MODULE_FILES["leads"])]
    contacts = [_extract_person_record(item) for item in load_json(RAW_MODULE_FILES["contacts"])]

    stats = BuildStats()
    stats.leads = _replace_people_table(cursor, "leads", leads)
    stats.contacts = _replace_people_table(cursor, "contacts", contacts)

    interaction_rows: list[tuple[Any, ...]] = []

    stats.notes, note_interactions = _load_notes(cursor, load_json(RAW_MODULE_FILES["notes"]))
    stats.calls, call_interactions = _load_calls(cursor, load_json(RAW_MODULE_FILES["calls"]))
    stats.tasks, task_interactions = _load_tasks(cursor, load_json(RAW_MODULE_FILES["tasks"]))
    stats.events, event_interactions = _load_events(cursor, load_json(RAW_MODULE_FILES["events"]))

    interaction_rows.extend(note_interactions)
    interaction_rows.extend(call_interactions)
    interaction_rows.extend(task_interactions)
    interaction_rows.extend(event_interactions)

    stats.interactions = _rebuild_interactions(cursor, interaction_rows)

    conn.commit()
    conn.close()
    return stats
