import sqlite3
from typing import Any

from assistant_core.config import WAREHOUSE_DB


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(WAREHOUSE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def dashboard_metrics(owner: str | None = None, date_from: str | None = None, date_to: str | None = None, status: str | None = None) -> dict[str, Any]:
    with _connect() as conn:
        filters = []
        params: list[Any] = []

        if owner:
            filters.append("owner_name = ?")
            params.append(owner)
        if date_from:
            filters.append("date(interaction_at) >= date(?)")
            params.append(date_from)
        if date_to:
            filters.append("date(interaction_at) <= date(?)")
            params.append(date_to)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        total_interactions = conn.execute(
            f"SELECT COUNT(*) AS total FROM interactions {where_clause}",
            params,
        ).fetchone()["total"]

        by_type = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT source_type, COUNT(*) AS total
                FROM interactions
                {where_clause}
                GROUP BY source_type
                ORDER BY total DESC
                """,
                params,
            ).fetchall()
        ]

        owner_load = [
            dict(row)
            for row in conn.execute(
                """
                SELECT owner_name, COUNT(*) AS total_records
                FROM (
                    SELECT owner_name FROM leads
                    UNION ALL
                    SELECT owner_name FROM contacts
                )
                WHERE owner_name IS NOT NULL
                GROUP BY owner_name
                ORDER BY total_records DESC
                LIMIT 10
                """
            ).fetchall()
        ]

        task_filters = []
        task_params: list[Any] = []
        if owner:
            task_filters.append("owner_name = ?")
            task_params.append(owner)
        if status:
            task_filters.append("lower(status) = lower(?)")
            task_params.append(status)
        if date_from:
            task_filters.append("date(COALESCE(due_date, created_time)) >= date(?)")
            task_params.append(date_from)
        if date_to:
            task_filters.append("date(COALESCE(due_date, created_time)) <= date(?)")
            task_params.append(date_to)

        task_where = f"WHERE {' AND '.join(task_filters)}" if task_filters else ""

        pending_tasks = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT owner_name, contact_name, subject, status, priority, due_date
                FROM tasks
                {task_where}
                ORDER BY due_date IS NULL, due_date ASC
                LIMIT 12
                """,
                task_params,
            ).fetchall()
        ]

        stale_contacts = [
            dict(row)
            for row in conn.execute(
                """
                WITH latest AS (
                    SELECT related_id, related_name, owner_name, MAX(interaction_at) AS last_touch
                    FROM interactions
                    WHERE related_id IS NOT NULL
                    GROUP BY related_id, related_name, owner_name
                )
                SELECT related_name, owner_name, last_touch
                FROM latest
                WHERE julianday('now') - julianday(last_touch) >= 30
                ORDER BY last_touch ASC
                LIMIT 12
                """
            ).fetchall()
        ]

        recent_activity = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT related_name, owner_name, source_type, interaction_at, status, summary
                FROM interactions
                {where_clause}
                ORDER BY interaction_at DESC
                LIMIT 12
                """,
                params,
            ).fetchall()
        ]

    return {
        "filters": {
            "owner": owner,
            "date_from": date_from,
            "date_to": date_to,
            "status": status,
        },
        "total_interactions": total_interactions,
        "by_type": by_type,
        "owner_load": owner_load,
        "pending_tasks": pending_tasks,
        "stale_contacts": stale_contacts,
        "recent_activity": recent_activity,
    }


def available_owners() -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT owner_name
            FROM (
                SELECT owner_name FROM leads
                UNION
                SELECT owner_name FROM contacts
                UNION
                SELECT owner_name FROM interactions
            )
            WHERE owner_name IS NOT NULL AND trim(owner_name) <> ''
            ORDER BY owner_name
            """
        ).fetchall()
    return [row[0] for row in rows]
