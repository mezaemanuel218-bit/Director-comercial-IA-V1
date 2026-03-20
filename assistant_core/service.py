import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from assistant_core.auth import AppUser
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

    def answer_question(self, question: str, user: AppUser | None = None) -> AssistantResponse:
        intent = classify_question(question)
        owner_scope = self._resolve_owner_scope(question, user)
        effective_question = self._normalize_user_scoped_question(question, owner_scope)
        evidence = self._collect_evidence(effective_question, intent, owner_scope=owner_scope)
        evidence["active_user"] = self._user_context(user)
        evidence["owner_scope"] = owner_scope

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

        prompt = self._build_prompt(effective_question, intent, evidence)
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

    def _collect_evidence(self, question: str, intent: QuestionIntent, owner_scope: str | None = None) -> dict[str, Any]:
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
                if intent.asks_for_names:
                    evidence["direct_answer"] = self._contact_points_for_entity(conn, entity_term)
                else:
                    evidence["direct_answer"] = self._emails_for_entity(conn, entity_term)
            elif intent.asks_for_phones and entity_term:
                evidence["direct_answer"] = self._phones_for_entity(conn, entity_term)
            elif intent.asks_for_owner_load:
                evidence["owner_load"] = self._owner_load(conn)
                if not evidence["owner_load"]:
                    evidence["owner_load"] = self._owner_load_from_interactions(conn)
                evidence["direct_answer"] = self._format_owner_load(evidence["owner_load"])
            elif intent.asks_for_interactions_by_owner:
                evidence["interactions_by_owner"] = self._interactions_by_owner(conn, owner_scope)
                evidence["direct_answer"] = self._format_interactions_by_owner(evidence["interactions_by_owner"], owner_scope)
            elif intent.asks_for_recent_activity_by_owner:
                evidence["recent_activity_by_owner"] = self._recent_activity_by_owner(conn, owner_scope)
                evidence["direct_answer"] = self._format_recent_activity_by_owner(evidence["recent_activity_by_owner"], owner_scope)
            elif intent.asks_for_yesterday_contacts:
                evidence["yesterday_contacts"] = self._interactions_on_relative_day(conn, 1, owner_scope)
                evidence["direct_answer"] = self._format_interaction_list(
                    evidence["yesterday_contacts"],
                    "No encontre contactos registrados ayer.",
                )
            elif intent.asks_for_day_before_yesterday_contacts:
                evidence["day_before_yesterday_contacts"] = self._interactions_on_relative_day(conn, 2, owner_scope)
                evidence["direct_answer"] = self._format_interaction_list(
                    evidence["day_before_yesterday_contacts"],
                    "No encontre contactos registrados antier.",
                )
            elif intent.asks_for_latest_contacted:
                evidence["latest_contacted"] = self._latest_contacted(conn)
                evidence["direct_answer"] = self._format_latest_contacted(evidence["latest_contacted"])
            elif intent.asks_for_today_pending:
                evidence["today_pending"] = self._today_pending(conn, owner_scope)
                evidence["direct_answer"] = self._format_today_pending(evidence["today_pending"])
            elif intent.asks_for_pending_commitments:
                evidence["pending_tasks"] = self._pending_tasks(conn, owner_scope)
                if intent.mode == "data":
                    evidence["direct_answer"] = self._format_pending_tasks(evidence["pending_tasks"])
            elif intent.asks_for_stale_contacts:
                evidence["stale_contacts"] = self._stale_contacts(conn, owner_scope)
                if intent.mode == "data":
                    evidence["direct_answer"] = self._format_stale_contacts(evidence["stale_contacts"])
            elif intent.asks_for_today_call_list:
                evidence["today_call_list"] = self.get_priority_followups(owner=owner_scope, limit=15)
                if intent.mode == "data":
                    evidence["direct_answer"] = self._format_today_call_list(evidence["today_call_list"])
            elif intent.asks_for_comparison:
                evidence["comparison_candidates"] = self._comparison_candidates(conn, question)
                evidence["direct_answer"] = self._format_comparison_candidates(evidence["comparison_candidates"])
            elif intent.asks_for_last_contact:
                evidence["last_interactions"] = self._last_interactions(conn, entity_term)
                if intent.mode == "data":
                    evidence["direct_answer"] = self._format_last_interactions(evidence["last_interactions"])
            elif intent.asks_for_kpis:
                evidence["owner_kpis"] = self._owner_kpis(conn, owner_scope)
                evidence["direct_answer"] = self._format_owner_kpis(evidence["owner_kpis"], owner_scope)
            elif intent.asks_for_risks and entity_term:
                evidence["risk_profile"] = self._risk_profile(conn, entity_term)
                evidence["direct_answer"] = self._format_risk_profile(evidence["risk_profile"], entity_term)

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

    def _user_context(self, user: AppUser | None) -> dict[str, Any] | None:
        if not user:
            return None
        return {
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
            "crm_owner_name": user.crm_owner_name,
        }

    def _resolve_owner_scope(self, question: str, user: AppUser | None) -> str | None:
        normalized = self._normalize_search_text(question)
        aliases = {
            "eduardo": "Eduardo Valdez",
            "evaldez": "Eduardo Valdez",
            "ceo": "Eduardo Valdez",
            "pablo": "Pablo Melin Dorador",
            "pmelin": "Pablo Melin Dorador",
            "emmanuel": "Jesus Emmanuel Meza Guzmán",
            "emeza": "Jesus Emmanuel Meza Guzmán",
            "jesus emmanuel meza": "Jesus Emmanuel Meza Guzmán",
            "jesus emmanuel meza guzman": "Jesus Emmanuel Meza Guzmán",
        }
        for alias, owner_name in aliases.items():
            if alias in normalized:
                return owner_name

        self_scope_signals = [" mi ", " mis ", " conmigo ", " yo ", "mias", "mios", "mías", "míos", " debo ", " tengo ", " hable ", " hablé "]
        padded = f" {normalized} "
        if user and any(signal in padded for signal in self_scope_signals):
            return user.crm_owner_name
        if user and user.role == "seller":
            seller_default_signals = [
                "a quien debo llamar hoy",
                "a quién debo llamar hoy",
                "a quien le hable ayer",
                "a quién le hablé ayer",
                "a quien le hable antier",
                "a quién le hablé antier",
                "compromisos pendientes para hoy",
                "kpi",
            ]
            if any(signal in normalized for signal in seller_default_signals):
                return user.crm_owner_name
        return None

    def _normalize_user_scoped_question(self, question: str, owner_scope: str | None) -> str:
        if not owner_scope:
            return question
        if any(token in question.lower() for token in [" mi ", " mis ", " yo ", "mias", "mios", "mías", "míos"]):
            return f"{question.strip()} del vendedor {owner_scope}"
        return question

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
        note_emails = self._emails_from_notes(conn, term)
        rows = sorted({*rows, *note_emails})
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

    def _contact_points_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        matches = self._entity_matches(conn, term)
        rows: list[str] = []
        seen: set[str] = set()

        for match in matches:
            name = match.get("contact_name") or match.get("full_name") or match.get("company_name") or "Sin nombre"
            email = match.get("email")
            if not email:
                continue
            key = f"{name.lower()}|{email.lower()}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(f"{name} | {email}")

        note_contacts = self._emails_from_notes(conn, term, include_names=True)
        for item in note_contacts:
            if item.lower() in seen:
                continue
            seen.add(item.lower())
            rows.append(item)

        if rows:
            return "\n".join(rows)
        return "No encontre nombres y correos para ese cliente o prospecto."

    def _emails_from_notes(self, conn: sqlite3.Connection, term: str, include_names: bool = False) -> list[str]:
        notes = self._recent_notes_for_entity(conn, term)
        extracted: list[str] = []
        seen: set[str] = set()
        for note in notes:
            content = note.get("content_text") or ""
            emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", content)
            for email in emails:
                cleaned_email = email.strip().strip(".,;:()[]<>").lower()
                if cleaned_email in seen:
                    continue
                seen.add(cleaned_email)
                if include_names:
                    name = self._guess_contact_name_from_note(content, cleaned_email) or "Contacto en nota"
                    extracted.append(f"{name} | {cleaned_email}")
                else:
                    extracted.append(cleaned_email)
        return extracted

    def _guess_contact_name_from_note(self, content: str, email: str) -> str | None:
        patterns = [
            rf"contacto correcto de ([A-Za-zÁÉÍÓÚÑáéíóúñ ]+) \([^)]*\).*?{re.escape(email)}",
            rf"correo de ([A-Za-zÁÉÍÓÚÑáéíóúñ ]+).*?{re.escape(email)}",
            rf"con ([A-Za-zÁÉÍÓÚÑáéíóúñ ]+) en copia.*?{re.escape(email)}",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return " ".join(match.group(1).split())
        local_name = email.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
        if local_name:
            return " ".join(part.capitalize() for part in local_name.split())
        return None

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

    def _owner_load_from_interactions(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, COUNT(DISTINCT related_id) AS total_records
        FROM interactions
        WHERE owner_name IS NOT NULL AND related_id IS NOT NULL
        GROUP BY owner_name
        ORDER BY total_records DESC, owner_name ASC
        """
        return [dict(row) for row in conn.execute(query).fetchall()]

    def _interactions_by_owner(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, COUNT(*) AS total_interactions
        FROM interactions
        WHERE owner_name IS NOT NULL
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " GROUP BY owner_name ORDER BY total_interactions DESC, owner_name ASC"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _recent_activity_by_owner(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, COUNT(*) AS recent_interactions, MAX(interaction_at) AS latest_activity
        FROM interactions
        WHERE owner_name IS NOT NULL
          AND interaction_at IS NOT NULL
          AND date(interaction_at) >= date('now', '-30 day')
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " GROUP BY owner_name ORDER BY recent_interactions DESC, latest_activity DESC"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _owner_kpis(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> dict[str, Any]:
        lead_where = "WHERE owner_name = ?" if owner_scope else ""
        contact_where = "WHERE owner_name = ?" if owner_scope else ""
        interaction_where = "WHERE owner_name = ?" if owner_scope else ""
        note_where = "WHERE owner_name = ?" if owner_scope else ""
        params = [owner_scope] if owner_scope else []

        leads = conn.execute(f"SELECT COUNT(*) FROM leads {lead_where}", params).fetchone()[0]
        contacts = conn.execute(f"SELECT COUNT(*) FROM contacts {contact_where}", params).fetchone()[0]
        notes = conn.execute(f"SELECT COUNT(*) FROM notes {note_where}", params).fetchone()[0]
        interactions = conn.execute(f"SELECT COUNT(*) FROM interactions {interaction_where}", params).fetchone()[0]
        last_activity_row = conn.execute(
            f"SELECT MAX(interaction_at) FROM interactions {interaction_where}",
            params,
        ).fetchone()
        return {
            "owner_scope": owner_scope,
            "leads": leads,
            "contacts": contacts,
            "notes": notes,
            "interactions": interactions,
            "last_activity": last_activity_row[0] if last_activity_row else None,
        }

    def _risk_profile(self, conn: sqlite3.Connection, term: str) -> dict[str, Any]:
        matches = self._entity_matches(conn, term)
        notes = self._recent_notes_for_entity(conn, term)
        interactions = self._recent_interactions_for_entity(conn, term)
        combined = " || ".join((note.get("content_text") or "").lower() for note in notes)
        risks: list[str] = []
        if "no cree" in combined or "rechaz" in combined or "no se acepte" in combined:
            risks.append("Hay resistencia comercial o rechazo parcial en notas.")
        if "reboto" in combined or "rebotó" in combined:
            risks.append("Hay correos previos rebotados o contactos incorrectos.")
        if "espera de respuesta" in combined or "sin respuesta" in combined:
            risks.append("Existe riesgo de enfriamiento por falta de respuesta.")
        if interactions:
            latest_touch = interactions[0].get("interaction_at")
        else:
            latest_touch = None
            risks.append("No hay interacciones registradas recientes.")
        if matches and not any(match.get("email") for match in matches):
            risks.append("El registro principal no tiene correo cargado directamente en CRM.")
        return {
            "matches": matches,
            "notes": notes,
            "interactions": interactions,
            "latest_touch": latest_touch,
            "risks": risks,
        }

    def _pending_tasks(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, contact_name, subject, status, priority, due_date, description
        FROM tasks
        WHERE status IS NULL OR lower(status) NOT IN ('completed', 'cancelled')
        ORDER BY due_date IS NULL, due_date ASC, owner_name ASC
        LIMIT 25
        """
        if owner_scope:
            query = """
            SELECT owner_name, contact_name, subject, status, priority, due_date, description
            FROM tasks
            WHERE (status IS NULL OR lower(status) NOT IN ('completed', 'cancelled'))
              AND owner_name = ?
            ORDER BY due_date IS NULL, due_date ASC, owner_name ASC
            LIMIT 25
            """
            return [dict(row) for row in conn.execute(query, (owner_scope,)).fetchall()]
        return [dict(row) for row in conn.execute(query).fetchall()]

    def _stale_contacts(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
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
        if owner_scope:
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
              AND owner_name = ?
            ORDER BY last_touch ASC
            LIMIT 25
            """
            return [dict(row) for row in conn.execute(query, (owner_scope,)).fetchall()]
        return [dict(row) for row in conn.execute(query).fetchall()]

    def _today_pending(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, contact_name, subject, status, priority, due_date, description
        FROM tasks
        WHERE date(due_date) = date('now', 'localtime')
          AND (status IS NULL OR lower(status) NOT IN ('completed', 'cancelled'))
        ORDER BY priority DESC, owner_name ASC
        LIMIT 25
        """
        if owner_scope:
            query = """
            SELECT owner_name, contact_name, subject, status, priority, due_date, description
            FROM tasks
            WHERE date(due_date) = date('now', 'localtime')
              AND (status IS NULL OR lower(status) NOT IN ('completed', 'cancelled'))
              AND owner_name = ?
            ORDER BY priority DESC, owner_name ASC
            LIMIT 25
            """
            return [dict(row) for row in conn.execute(query, (owner_scope,)).fetchall()]
        return [dict(row) for row in conn.execute(query).fetchall()]

    def _interactions_on_relative_day(self, conn: sqlite3.Connection, days_ago: int, owner_scope: str | None = None) -> list[dict[str, Any]]:
        modifier = f"-{days_ago} day"
        query = """
        SELECT related_name, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE interaction_at IS NOT NULL
          AND date(interaction_at) = date('now', ?, 'localtime')
        ORDER BY interaction_at DESC
        LIMIT 25
        """
        if owner_scope:
            query = """
            SELECT related_name, owner_name, source_type, interaction_at, status, summary
            FROM interactions
            WHERE interaction_at IS NOT NULL
              AND date(interaction_at) = date('now', ?, 'localtime')
              AND owner_name = ?
            ORDER BY interaction_at DESC
            LIMIT 25
            """
            return [dict(row) for row in conn.execute(query, (modifier, owner_scope)).fetchall()]
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
            signals = self._commercial_signals(notes)
            results.append({
                "term": term,
                "matches": matches[:3],
                "recent_interactions": recent[:5],
                "recent_notes": notes[:4],
                "latest_touch": latest_touch,
                "owners": owner_names,
                "interaction_count": len(recent),
                "signals": signals,
                "advance_score": self._advance_score(matches, recent, notes, signals),
            })
        return results

    def _commercial_signals(self, notes: list[dict[str, Any]]) -> list[str]:
        signal_keywords = {
            "demo": "Hay solicitud o referencia de demo.",
            "cotiz": "Hay referencia a cotizacion.",
            "seguimiento": "Hay seguimiento documentado.",
            "interes": "Hay senal de interes.",
            "prueba": "Hay referencia a prueba.",
            "visita": "Hay visita registrada.",
            "correo": "Hay correo documentado.",
            "llamad": "Hay llamada documentada.",
            "reunion": "Hay reunion mencionada.",
            "propuesta": "Hay propuesta mencionada.",
        }
        combined = " || ".join((note.get("content_text") or "").lower() for note in notes)
        found: list[str] = []
        for keyword, description in signal_keywords.items():
            if keyword in combined:
                found.append(description)
        return found[:5]

    def _advance_score(
        self,
        matches: list[dict[str, Any]],
        recent_interactions: list[dict[str, Any]],
        notes: list[dict[str, Any]],
        signals: list[str],
    ) -> int:
        score = 0
        if matches:
            score += 10
        if any(match.get("owner_name") for match in matches):
            score += 10
        score += min(20, len(recent_interactions) * 4)
        score += min(20, len(notes) * 3)
        score += min(20, len(signals) * 4)
        combined = " || ".join((note.get("content_text") or "").lower() for note in notes)
        if "rechaz" in combined or "no cree" in combined:
            score -= 8
        if "espera de respuesta" in combined or "pendiente" in combined:
            score += 4
        return score

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

    def _format_interactions_by_owner(self, rows: list[dict[str, Any]], owner_scope: str | None = None) -> str:
        if not rows:
            return "No hay evidencia disponible sobre la cantidad de interacciones que tiene cada vendedor."
        if owner_scope and rows:
            top = rows[0]
            return f"{top['owner_name']} tiene {top['total_interactions']} interacciones registradas."
        return "\n".join(f"{row['owner_name']}: {row['total_interactions']}" for row in rows)

    def _format_recent_activity_by_owner(self, rows: list[dict[str, Any]], owner_scope: str | None = None) -> str:
        if not rows:
            return "No hay evidencia disponible para determinar quién tiene más actividad reciente."
        if owner_scope:
            top = rows[0]
            return (
                f"{top['owner_name']} tiene {top['recent_interactions']} interacciones recientes "
                f"en los ultimos 30 dias. Ultima actividad: {top['latest_activity'] or 'sin dato'}."
            )
        top = rows[0]
        lines = [f"Con mas actividad reciente: {top['owner_name']} ({top['recent_interactions']})", ""]
        lines.extend(
            f"{row['owner_name']}: {row['recent_interactions']} | ultima actividad {row['latest_activity'] or 'sin dato'}"
            for row in rows
        )
        return "\n".join(lines)

    def _format_owner_kpis(self, kpis: dict[str, Any], owner_scope: str | None = None) -> str:
        target = owner_scope or "todo el equipo"
        return (
            f"KPIs de {target}:\n"
            f"- Leads: {kpis['leads']}\n"
            f"- Contacts: {kpis['contacts']}\n"
            f"- Notas: {kpis['notes']}\n"
            f"- Interacciones: {kpis['interactions']}\n"
            f"- Ultima actividad registrada: {kpis['last_activity'] or 'sin dato'}"
        )

    def _format_risk_profile(self, risk_profile: dict[str, Any], term: str) -> str:
        if not risk_profile.get("matches") and not risk_profile.get("notes") and not risk_profile.get("interactions"):
            return f"No hay evidencia disponible sobre el prospecto {term}. Por lo tanto, no puedo identificar riesgos específicos."
        risks = risk_profile.get("risks") or ["No se detectan riesgos claros con la evidencia disponible."]
        latest_touch = risk_profile.get("latest_touch") or "Sin interaccion registrada"
        return (
            f"Riesgos identificados para {term}:\n"
            f"- Ultimo toque: {latest_touch}\n"
            + "\n".join(f"- {risk}" for risk in risks)
        )

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
        scored_candidates: list[dict[str, Any]] = []
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

            signals = candidate.get("signals") or []

            formatted_sections.append(
                "\n".join(
                    [
                        entity_name,
                        f"- Propietario(s): {owners}",
                        f"- Ultimo toque: {candidate.get('latest_touch') or 'Sin interaccion registrada'}",
                        f"- Interacciones recientes detectadas: {candidate.get('interaction_count', 0)}",
                        f"- Correo: {email}",
                        f"- Telefono: {phone}",
                        f"- Senales comerciales: {', '.join(signals) if signals else 'Sin senales claras'}",
                        f"- Nota reciente: {note_summary}",
                    ]
                )
            )
            scored_candidates.append(
                {
                    "name": entity_name,
                    "score": candidate.get("advance_score", 0),
                    "signals": signals,
                    "latest_touch": candidate.get("latest_touch"),
                    "owners": owners,
                }
            )

        if valid_candidates == 0:
            return None

        scored_candidates.sort(key=lambda item: item["score"], reverse=True)
        leader = scored_candidates[0]
        trailer = scored_candidates[-1]

        closing = (
            f"Conclusion:\n{leader['name']} va mas avanzado que {trailer['name']} "
            f"porque acumula mas evidencia comercial util en notas, interacciones y asignacion de propietario "
            f"(score {leader['score']} vs {trailer['score']})."
        )
        if leader["signals"]:
            closing += f" Las senales mas claras son: {', '.join(leader['signals'][:3])}."

        return "Comparativa encontrada:\n\n" + "\n\n".join(formatted_sections) + "\n\n" + closing

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
- Si hay usuario activo, interpreta las preguntas en primera persona segun su perfil salvo que se mencione otro vendedor explicitamente.

Modo detectado: {intent.mode}
Pregunta: {question}

Evidencia:
{evidence}
        """.strip()
