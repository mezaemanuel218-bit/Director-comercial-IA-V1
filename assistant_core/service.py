import os
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from assistant_core.config import WAREHOUSE_DB
from assistant_core.query_intent import QuestionIntent, classify_question


load_dotenv()


@dataclass
class AssistantResponse:
    mode: str
    sources: list[str]
    used_web: bool
    answer: str
    evidence: dict[str, Any]


class SalesAssistantService:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or str(WAREHOUSE_DB)
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def answer_question(self, question: str) -> AssistantResponse:
        intent = classify_question(question)
        evidence = self._collect_evidence(question, intent)

        if evidence.get("direct_answer") and intent.mode in {"data", "analysis"}:
            return AssistantResponse(
                mode=intent.mode,
                sources=evidence["sources"],
                used_web=False,
                answer=evidence["direct_answer"],
                evidence=evidence,
            )

        if not self.client:
            fallback = evidence.get("direct_answer") or self._format_evidence_fallback(evidence)
            return AssistantResponse(
                mode=intent.mode,
                sources=evidence["sources"],
                used_web=False,
                answer=fallback,
                evidence=evidence,
            )

        prompt = self._build_prompt(question, intent, evidence)
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return AssistantResponse(
            mode=intent.mode,
            sources=evidence["sources"],
            used_web=False,
            answer=response.choices[0].message.content.strip(),
            evidence=evidence,
        )

    def get_priority_followups(self, owner: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
        with self._connect() as conn:
            latest_query = """
            WITH latest AS (
                SELECT
                    related_id,
                    related_name,
                    owner_name,
                    MAX(interaction_at) AS last_touch
                FROM interactions
                WHERE related_id IS NOT NULL
                GROUP BY related_id, related_name, owner_name
            ),
            pending AS (
                SELECT
                    contact_id AS related_id,
                    COUNT(*) AS pending_count,
                    MIN(due_date) AS next_due_date
                FROM tasks
                WHERE contact_id IS NOT NULL
                  AND (status IS NULL OR lower(status) NOT IN ('completed', 'cancelled'))
                GROUP BY contact_id
            ),
            notes_signal AS (
                SELECT
                    parent_id AS related_id,
                    MAX(created_time) AS last_note_time,
                    GROUP_CONCAT(lower(content_text), ' || ') AS notes_text
                FROM notes
                WHERE parent_id IS NOT NULL
                GROUP BY parent_id
            )
            SELECT
                latest.related_id,
                latest.related_name,
                latest.owner_name,
                latest.last_touch,
                CAST(julianday('now') - julianday(latest.last_touch) AS INTEGER) AS days_without_contact,
                COALESCE(pending.pending_count, 0) AS pending_count,
                pending.next_due_date,
                notes_signal.last_note_time,
                notes_signal.notes_text
            FROM latest
            LEFT JOIN pending ON pending.related_id = latest.related_id
            LEFT JOIN notes_signal ON notes_signal.related_id = latest.related_id
            """
            params: list[Any] = []
            if owner:
                latest_query += " WHERE latest.owner_name = ?"
                params.append(owner)
            latest_query += " ORDER BY days_without_contact DESC, pending_count DESC LIMIT ?"
            params.append(limit * 3)

            rows = [dict(row) for row in conn.execute(latest_query, params).fetchall()]

        prioritized: list[dict[str, Any]] = []
        for row in rows:
            score = 0
            reasons: list[str] = []
            days_without_contact = row.get("days_without_contact") or 0
            pending_count = row.get("pending_count") or 0
            notes_text = row.get("notes_text") or ""

            if days_without_contact >= 30:
                score += 40
                reasons.append("Tiene 30 o mas dias sin contacto.")
            elif days_without_contact >= 14:
                score += 24
                reasons.append("Tiene mas de 14 dias sin seguimiento.")
            elif days_without_contact >= 7:
                score += 12
                reasons.append("Tiene una semana o mas sin movimiento.")

            if pending_count > 0:
                score += min(25, pending_count * 8)
                reasons.append(f"Tiene {pending_count} compromiso(s) pendiente(s).")

            signal_keywords = {
                "demo": 12,
                "cotiz": 10,
                "reuni": 9,
                "seguimiento": 8,
                "llamar": 8,
                "instal": 7,
                "prueba": 10,
                "permisionario": 6,
                "interes": 9,
                "interés": 9,
            }
            for keyword, points in signal_keywords.items():
                if keyword in notes_text:
                    score += points
                    reasons.append(f"Notas con señal comercial: '{keyword}'.")
                    break

            if "rechaz" in notes_text or "cancel" in notes_text:
                score -= 8
                reasons.append("Hay nota de rechazo o cancelacion reciente.")

            if score <= 0:
                continue

            prioritized.append(
                {
                    "related_id": row["related_id"],
                    "related_name": row["related_name"],
                    "owner_name": row["owner_name"],
                    "last_touch": row["last_touch"],
                    "days_without_contact": days_without_contact,
                    "pending_count": pending_count,
                    "score": score,
                    "priority_label": self._priority_label(score),
                    "reasons": reasons[:3],
                }
            )

        prioritized.sort(key=lambda item: (-item["score"], -item["days_without_contact"], item["related_name"] or ""))
        return prioritized[:limit]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _collect_evidence(self, question: str, intent: QuestionIntent) -> dict[str, Any]:
        entity_term = self._extract_entity_hint(question)
        evidence: dict[str, Any] = {
            "sources": ["warehouse.db"],
            "entity_hint": entity_term,
            "direct_answer": None,
        }

        with self._connect() as conn:
            if entity_term:
                evidence["matches"] = self._entity_matches(conn, entity_term)
                evidence["recent_interactions"] = self._recent_interactions_for_entity(conn, entity_term)
                evidence["recent_notes"] = self._recent_notes_for_entity(conn, entity_term)

            if intent.asks_for_emails and entity_term:
                evidence["direct_answer"] = self._emails_for_entity(conn, entity_term)
            elif intent.asks_for_phones and entity_term:
                evidence["direct_answer"] = self._phones_for_entity(conn, entity_term)
            elif intent.asks_for_owner_load:
                evidence["owner_load"] = self._owner_load(conn)
                evidence["direct_answer"] = self._format_owner_load(evidence["owner_load"])
            elif intent.asks_for_yesterday_contacts:
                evidence["yesterday_contacts"] = self._interactions_on_relative_day(conn, 1)
                evidence["direct_answer"] = self._format_interaction_list(
                    evidence["yesterday_contacts"],
                    "No encontre contactos registrados ayer.",
                )
            elif intent.asks_for_day_before_yesterday_contacts:
                evidence["day_before_yesterday_contacts"] = self._interactions_on_relative_day(conn, 2)
                evidence["direct_answer"] = self._format_interaction_list(
                    evidence["day_before_yesterday_contacts"],
                    "No encontre contactos registrados antier.",
                )
            elif intent.asks_for_latest_contacted:
                evidence["latest_contacted"] = self._latest_contacted(conn)
                evidence["direct_answer"] = self._format_latest_contacted(evidence["latest_contacted"])
            elif intent.asks_for_today_pending:
                evidence["today_pending"] = self._today_pending(conn)
                evidence["direct_answer"] = self._format_today_pending(evidence["today_pending"])
            elif intent.asks_for_pending_commitments:
                evidence["pending_tasks"] = self._pending_tasks(conn)
                if intent.mode == "data":
                    evidence["direct_answer"] = self._format_pending_tasks(evidence["pending_tasks"])
            elif intent.asks_for_stale_contacts:
                evidence["stale_contacts"] = self._stale_contacts(conn)
                if intent.mode == "data":
                    evidence["direct_answer"] = self._format_stale_contacts(evidence["stale_contacts"])
            elif intent.asks_for_today_call_list:
                evidence["today_call_list"] = self.get_priority_followups(owner=None, limit=15)
                if intent.mode == "data":
                    evidence["direct_answer"] = self._format_today_call_list(evidence["today_call_list"])
            elif intent.asks_for_comparison:
                evidence["comparison_candidates"] = self._comparison_candidates(conn, question)
                evidence["direct_answer"] = self._format_comparison_candidates(evidence["comparison_candidates"])
            elif intent.asks_for_last_contact:
                evidence["last_interactions"] = self._last_interactions(conn, entity_term)
                if intent.mode == "data":
                    evidence["direct_answer"] = self._format_last_interactions(evidence["last_interactions"])

            evidence["document_chunks"] = self._document_search(conn, question)

        return evidence

    def _extract_entity_hint(self, question: str) -> str | None:
        prefixes = [
            "de ",
            "del ",
            "para ",
            "cliente ",
            "prospecto ",
            "contacto ",
        ]
        lower_question = question.lower()
        for prefix in prefixes:
            position = lower_question.find(prefix)
            if position >= 0:
                return self._clean_search_term(question[position + len(prefix):].strip(" ?."))
        return None

    def _entity_matches(self, conn: sqlite3.Connection, term: str) -> list[dict[str, Any]]:
        cleaned_term = self._clean_search_term(term)
        normalized_term = self._normalize_search_text(cleaned_term)
        query = """
        SELECT 'lead' AS entity_type, id, company_name, contact_name, full_name, email, phone, owner_name, last_activity_time
        FROM leads
        UNION ALL
        SELECT 'contact' AS entity_type, id, company_name, contact_name, full_name, email, phone, owner_name, last_activity_time
        FROM contacts
        """
        rows = [dict(row) for row in conn.execute(query).fetchall()]
        scored: list[tuple[int, dict[str, Any]]] = []

        for row in rows:
            fields = [
                row.get("company_name") or "",
                row.get("contact_name") or "",
                row.get("full_name") or "",
            ]
            joined = " ".join(fields)
            normalized_joined = self._normalize_search_text(joined)
            if not normalized_joined:
                continue

            score = 0
            if normalized_term and normalized_term in normalized_joined:
                score += 50

            tokens = [token for token in normalized_term.split() if token]
            token_hits = sum(1 for token in tokens if token in normalized_joined)
            score += token_hits * 10

            if row.get("company_name") and normalized_term == self._normalize_search_text(row["company_name"]):
                score += 40
            if row.get("contact_name") and normalized_term == self._normalize_search_text(row["contact_name"]):
                score += 30

            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda item: (-item[0], item[1].get("company_name") or item[1].get("full_name") or ""))
        return [row for _, row in scored[:10]]

    def _recent_interactions_for_entity(self, conn: sqlite3.Connection, term: str) -> list[dict[str, Any]]:
        matches = self._entity_matches(conn, term)
        related_ids = [match["id"] for match in matches if match.get("id")]
        if related_ids:
            placeholders = ", ".join("?" for _ in related_ids)
            query = f"""
            SELECT source_type, related_name, owner_name, interaction_at, status, summary
            FROM interactions
            WHERE related_id IN ({placeholders})
            ORDER BY interaction_at DESC
            LIMIT 10
            """
            return [dict(row) for row in conn.execute(query, related_ids).fetchall()]

        wildcard = f"%{self._clean_search_term(term)}%"
        query = """
        SELECT source_type, related_name, owner_name, interaction_at, status, summary
        FROM interactions
        WHERE lower(related_name) LIKE lower(?)
        ORDER BY interaction_at DESC
        LIMIT 10
        """
        return [dict(row) for row in conn.execute(query, (wildcard,)).fetchall()]

    def _recent_notes_for_entity(self, conn: sqlite3.Connection, term: str) -> list[dict[str, Any]]:
        matches = self._entity_matches(conn, term)
        parent_ids = [match["id"] for match in matches if match.get("id")]
        if parent_ids:
            placeholders = ", ".join("?" for _ in parent_ids)
            query = f"""
            SELECT parent_name, parent_module, created_time, title, content_text
            FROM notes
            WHERE parent_id IN ({placeholders})
            ORDER BY created_time DESC
            LIMIT 8
            """
            return [dict(row) for row in conn.execute(query, parent_ids).fetchall()]

        wildcard = f"%{self._clean_search_term(term)}%"
        query = """
        SELECT parent_name, parent_module, created_time, title, content_text
        FROM notes
        WHERE lower(parent_name) LIKE lower(?) OR lower(content_text) LIKE lower(?)
        ORDER BY created_time DESC
        LIMIT 8
        """
        return [dict(row) for row in conn.execute(query, (wildcard, wildcard)).fetchall()]

    def _emails_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        rows = self._field_values_for_entity(conn, term, "email")
        return "\n".join(rows) if rows else "No encontre correos para ese cliente o prospecto."

    def _phones_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        rows = self._field_values_for_entity(conn, term, "phone")
        return "\n".join(rows) if rows else "No encontre telefonos para ese cliente o prospecto."

    def _field_values_for_entity(self, conn: sqlite3.Connection, term: str, field_name: str) -> list[str]:
        matches = self._entity_matches(conn, term)
        values = [match.get(field_name) for match in matches if match.get(field_name)]
        deduplicated = sorted({value.strip() for value in values if value and value.strip()})
        if deduplicated:
            return deduplicated

        wildcard = f"%{self._clean_search_term(term)}%"
        query = f"""
        SELECT DISTINCT {field_name}
        FROM (
            SELECT {field_name}, company_name, contact_name, full_name FROM leads
            UNION ALL
            SELECT {field_name}, company_name, contact_name, full_name FROM contacts
        )
        WHERE {field_name} IS NOT NULL
          AND (
              lower(company_name) LIKE lower(?)
              OR lower(contact_name) LIKE lower(?)
              OR lower(full_name) LIKE lower(?)
          )
        ORDER BY {field_name}
        """
        return [row[0] for row in conn.execute(query, (wildcard, wildcard, wildcard)).fetchall()]

    def _owner_load(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, COUNT(*) AS total_records
        FROM (
            SELECT owner_name FROM leads
            UNION ALL
            SELECT owner_name FROM contacts
        )
        WHERE owner_name IS NOT NULL
        GROUP BY owner_name
        ORDER BY total_records DESC, owner_name ASC
        """
        return [dict(row) for row in conn.execute(query).fetchall()]

    def _pending_tasks(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, contact_name, subject, status, priority, due_date, description
        FROM tasks
        WHERE status IS NULL OR lower(status) NOT IN ('completed', 'cancelled')
        ORDER BY due_date IS NULL, due_date ASC, owner_name ASC
        LIMIT 25
        """
        return [dict(row) for row in conn.execute(query).fetchall()]

    def _stale_contacts(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        query = """
        WITH latest AS (
            SELECT related_id, related_name, related_module, owner_name, MAX(interaction_at) AS last_touch
            FROM interactions
            WHERE related_id IS NOT NULL
            GROUP BY related_id, related_name, related_module, owner_name
        )
        SELECT related_name, related_module, owner_name, last_touch
        FROM latest
        WHERE julianday('now') - julianday(last_touch) >= 30
        ORDER BY last_touch ASC
        LIMIT 25
        """
        return [dict(row) for row in conn.execute(query).fetchall()]

    def _today_pending(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, contact_name, subject, status, priority, due_date, description
        FROM tasks
        WHERE date(due_date) = date('now', 'localtime')
          AND (status IS NULL OR lower(status) NOT IN ('completed', 'cancelled'))
        ORDER BY priority DESC, owner_name ASC
        LIMIT 25
        """
        return [dict(row) for row in conn.execute(query).fetchall()]

    def _interactions_on_relative_day(self, conn: sqlite3.Connection, days_ago: int) -> list[dict[str, Any]]:
        modifier = f"-{days_ago} day"
        query = """
        SELECT related_name, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE interaction_at IS NOT NULL
          AND date(interaction_at) = date('now', ?, 'localtime')
        ORDER BY interaction_at DESC
        LIMIT 25
        """
        return [dict(row) for row in conn.execute(query, (modifier,)).fetchall()]

    def _latest_contacted(self, conn: sqlite3.Connection) -> dict[str, Any] | None:
        query = """
        SELECT related_name, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE related_name IS NOT NULL
        ORDER BY interaction_at DESC
        LIMIT 1
        """
        row = conn.execute(query).fetchone()
        return dict(row) if row else None

    def _today_call_list(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        query = """
        WITH latest AS (
            SELECT related_id, related_name, owner_name, MAX(interaction_at) AS last_touch
            FROM interactions
            WHERE related_id IS NOT NULL
            GROUP BY related_id, related_name, owner_name
        )
        SELECT related_name, owner_name, last_touch
        FROM latest
        WHERE julianday('now') - julianday(last_touch) >= 3
        ORDER BY last_touch ASC
        LIMIT 15
        """
        rows = [dict(row) for row in conn.execute(query).fetchall()]
        for row in rows:
            row["reason"] = "Tiene al menos 3 dias sin interaccion registrada."
        return rows

    def _priority_label(self, score: int) -> str:
        if score >= 50:
            return "alta"
        if score >= 25:
            return "media"
        return "normal"

    def _comparison_candidates(self, conn: sqlite3.Connection, question: str) -> list[dict[str, Any]]:
        separators = [" vs ", " VS ", " contra ", " versus ", ","]
        normalized = question
        for separator in separators:
            normalized = normalized.replace(separator, "|")
        terms = [part.strip(" ?.\"'") for part in normalized.split("|")]
        cleaned_terms = [self._clean_search_term(term) for term in terms if len(term.strip()) >= 3]
        results: list[dict[str, Any]] = []
        for term in cleaned_terms[:2]:
            matches = self._entity_matches(conn, term)
            recent = self._recent_interactions_for_entity(conn, term)
            notes = self._recent_notes_for_entity(conn, term)
            latest_touch = recent[0]["interaction_at"] if recent else None
            owner_names = sorted({match.get("owner_name") for match in matches if match.get("owner_name")})
            results.append({
                "term": term,
                "matches": matches[:3],
                "recent_interactions": recent[:5],
                "recent_notes": notes[:4],
                "latest_touch": latest_touch,
                "owners": owner_names,
                "interaction_count": len(recent),
            })
        return results

    def _normalize_search_text(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
        return " ".join(ascii_text.lower().split())

    def _clean_search_term(self, value: str) -> str:
        cleaned = value or ""
        removable_prefixes = [
            "compara ",
            "comparar ",
            "comparativa ",
            "dame ",
            "cliente ",
            "prospecto ",
            "contacto ",
            "empresa ",
        ]
        lowered = cleaned.lower()
        changed = True
        while changed:
            changed = False
            for prefix in removable_prefixes:
                if lowered.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    lowered = cleaned.lower()
                    changed = True
        return cleaned.strip()

    def _last_interactions(self, conn: sqlite3.Connection, term: str | None) -> list[dict[str, Any]]:
        if term:
            matches = self._entity_matches(conn, term)
            related_ids = [match["id"] for match in matches if match.get("id")]
            if related_ids:
                placeholders = ", ".join("?" for _ in related_ids)
                query = f"""
                SELECT related_name, owner_name, source_type, interaction_at, status, summary
                FROM interactions
                WHERE related_id IN ({placeholders})
                ORDER BY interaction_at DESC
                LIMIT 10
                """
                rows = conn.execute(query, related_ids).fetchall()
            else:
                wildcard = f"%{self._clean_search_term(term)}%"
                query = """
                SELECT related_name, owner_name, source_type, interaction_at, status, summary
                FROM interactions
                WHERE lower(related_name) LIKE lower(?)
                ORDER BY interaction_at DESC
                LIMIT 10
                """
                rows = conn.execute(query, (wildcard,)).fetchall()
        else:
            query = """
            SELECT related_name, owner_name, source_type, interaction_at, status, summary
            FROM interactions
            ORDER BY interaction_at DESC
            LIMIT 10
            """
            rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]

    def _document_search(self, conn: sqlite3.Connection, question: str) -> list[dict[str, Any]]:
        terms = [term for term in question.split() if len(term) >= 4][:6]
        if not terms:
            return []

        like_clauses = " OR ".join("lower(content) LIKE lower(?)" for _ in terms)
        query = f"""
        SELECT d.file_name, dc.chunk_index, dc.content
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE {like_clauses}
        LIMIT 6
        """
        values = [f"%{term}%" for term in terms]
        try:
            return [dict(row) for row in conn.execute(query, values).fetchall()]
        except sqlite3.OperationalError:
            return []

    def _format_owner_load(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre propietarios con registros."
        lines = [f"{row['owner_name']}: {row['total_records']}" for row in rows]
        top = rows[0]
        summary = f"Con mas clientes asignados: {top['owner_name']} ({top['total_records']})"
        return summary + "\n\n" + "\n".join(lines)

    def _format_pending_tasks(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre compromisos pendientes."
        return "\n".join(
            f"{row['owner_name']} | {row['contact_name'] or 'Sin contacto'} | {row['subject']} | {row['status'] or 'Sin estatus'} | {row['due_date'] or 'Sin fecha'}"
            for row in rows
        )

    def _format_today_pending(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre compromisos pendientes para hoy."
        return "\n".join(
            f"{row['owner_name']} | {row['contact_name'] or 'Sin contacto'} | {row['subject']} | {row['priority'] or 'Sin prioridad'}"
            for row in rows
        )

    def _format_stale_contacts(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre clientes con mas de 30 dias sin contacto."
        return "\n".join(
            f"{row['related_name']} | {row['related_module']} | {row['owner_name']} | {row['last_touch']}"
            for row in rows
        )

    def _format_interaction_list(self, rows: list[dict[str, Any]], empty_message: str) -> str:
        if not rows:
            return empty_message
        return "\n".join(
            f"{row['related_name']} | {row['source_type']} | {row['interaction_at']} | {row['owner_name']}"
            for row in rows
        )

    def _format_latest_contacted(self, row: dict[str, Any] | None) -> str:
        if not row:
            return "No encontre un ultimo contacto registrado."
        return f"{row['related_name']} | {row['source_type']} | {row['interaction_at']} | {row['owner_name']}"

    def _format_today_call_list(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre clientes sugeridos para llamar hoy."
        return "\n".join(
            f"{row['related_name']} | {row['owner_name']} | prioridad {row['priority_label']} | score {row['score']} | {' / '.join(row['reasons'])}"
            for row in rows
        )

    def _format_comparison_candidates(self, rows: list[dict[str, Any]]) -> str | None:
        if len(rows) < 2:
            return None

        formatted_sections: list[str] = []
        valid_candidates = 0
        for candidate in rows[:2]:
            top_match = candidate["matches"][0] if candidate.get("matches") else None
            recent_note = candidate["recent_notes"][0] if candidate.get("recent_notes") else None
            owners = ", ".join(candidate.get("owners") or []) or "Sin propietario identificado"

            if top_match:
                valid_candidates += 1
                entity_name = (
                    top_match.get("company_name")
                    or top_match.get("contact_name")
                    or top_match.get("full_name")
                    or candidate["term"]
                )
                email = top_match.get("email") or "Sin correo"
                phone = top_match.get("phone") or "Sin telefono"
            else:
                entity_name = candidate["term"]
                email = "Sin correo"
                phone = "Sin telefono"

            note_summary = "Sin notas recientes relevantes."
            if recent_note:
                note_summary = recent_note.get("content_text") or recent_note.get("title") or "Nota sin contenido visible."
                note_summary = " ".join(note_summary.split())[:180]

            formatted_sections.append(
                "\n".join(
                    [
                        entity_name,
                        f"- Propietario(s): {owners}",
                        f"- Ultimo toque: {candidate.get('latest_touch') or 'Sin interaccion registrada'}",
                        f"- Interacciones recientes detectadas: {candidate.get('interaction_count', 0)}",
                        f"- Correo: {email}",
                        f"- Telefono: {phone}",
                        f"- Nota reciente: {note_summary}",
                    ]
                )
            )

        if valid_candidates == 0:
            return None

        return "Comparativa encontrada:\n\n" + "\n\n".join(formatted_sections)

    def _format_last_interactions(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre interacciones recientes."
        return "\n".join(
            f"{row['related_name']} | {row['source_type']} | {row['interaction_at']} | {row['owner_name']}"
            for row in rows
        )

    def _format_evidence_fallback(self, evidence: dict[str, Any]) -> str:
        sections: list[str] = []
        for key in (
            "matches",
            "recent_interactions",
            "recent_notes",
            "pending_tasks",
            "today_pending",
            "today_call_list",
            "stale_contacts",
            "comparison_candidates",
            "latest_contacted",
        ):
            rows = evidence.get(key)
            if rows:
                sections.append(f"{key}: {rows}")
        return "\n\n".join(sections) if sections else "No encontre evidencia suficiente para responder."

    def _build_prompt(self, question: str, intent: QuestionIntent, evidence: dict[str, Any]) -> str:
        return f"""
Eres el asistente comercial del departamento de ventas de Flotimatics.

Reglas obligatorias:
- Usa solo la evidencia proporcionada.
- No inventes datos.
- Si falta evidencia, dilo claramente.
- Solo menciona web si realmente se uso. En este caso no se uso web.
- Si la pregunta es precisa, responde directo y sin relleno.
- Si la pregunta pide analisis, separa hechos, interpretacion y recomendacion.
- Si la pregunta trata de seguimiento o prioridad comercial, explica la razon concreta de la prioridad.
- Si comparas prospectos, contrasta actividad reciente, notas, responsable y señales de avance.

Modo detectado: {intent.mode}
Pregunta: {question}

Evidencia:
{evidence}
        """.strip()
