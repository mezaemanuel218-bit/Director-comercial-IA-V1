import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from assistant_core.auth import APP_USERS, AppUser
from assistant_core.config import WAREHOUSE_DB
from assistant_core.query_intent import QuestionIntent, classify_question
from assistant_core.utils import strip_html


load_dotenv()


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d[\d\s().-]{6,}\d))")


@dataclass
class AssistantResponse:
    mode: str
    sources: list[str]
    used_web: bool
    answer: str
    evidence: dict[str, Any]


@dataclass
class ContactRow:
    label: str
    company: str
    email: str
    phone: str
    owner_name: str
    source: str


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

        if evidence.get("direct_answer") and self._should_prefer_direct_answer(intent, evidence):
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
        content = (response.choices[0].message.content or "").strip()
        answer = content or evidence.get("direct_answer") or self._format_evidence_fallback(evidence)
        if self._response_looks_empty(answer) and evidence.get("direct_answer"):
            answer = evidence["direct_answer"]
        return AssistantResponse(
            mode=intent.mode,
            sources=evidence["sources"],
            used_web=False,
            answer=answer,
            evidence=evidence,
        )

    def get_priority_followups(self, owner: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
        with self._connect() as conn:
            query = """
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
                COALESCE(notes_signal.notes_text, '') AS notes_text
            FROM latest
            LEFT JOIN pending ON pending.related_id = latest.related_id
            LEFT JOIN notes_signal ON notes_signal.related_id = latest.related_id
            """
            params: list[Any] = []
            if owner:
                query += " WHERE latest.owner_name = ?"
                params.append(owner)
            query += " ORDER BY days_without_contact DESC, pending_count DESC LIMIT ?"
            params.append(limit * 3)
            rows = [dict(row) for row in conn.execute(query, params).fetchall()]

        prioritized: list[dict[str, Any]] = []
        for row in rows:
            score = 0
            reasons: list[str] = []
            days_without_contact = row.get("days_without_contact") or 0
            pending_count = row.get("pending_count") or 0
            notes_text = (row.get("notes_text") or "").lower()

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

            for keyword, points in {
                "demo": 12,
                "cotiz": 10,
                "reuni": 9,
                "seguimiento": 8,
                "llamar": 8,
                "instal": 7,
                "prueba": 10,
                "interes": 9,
            }.items():
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

    def _should_prefer_direct_answer(self, intent: QuestionIntent, evidence: dict[str, Any]) -> bool:
        if intent.mode == "data":
            return True
        structured_keys = [
            "today_call_list",
            "comparison_candidates",
            "owner_load",
            "owner_kpis",
            "global_kpis",
            "assigned_clients",
            "pending_tasks",
            "today_pending",
            "stale_contacts",
            "latest_contacted",
            "last_interactions",
            "risk_profile",
        ]
        return any(evidence.get(key) for key in structured_keys)

    def _response_looks_empty(self, answer: str) -> bool:
        normalized = self._normalize_search_text(answer)
        empty_markers = [
            "no hay registros",
            "no se encontraron registros",
            "no existe informacion",
            "no hay evidencia",
            "no es posible realizar",
        ]
        return any(marker in normalized for marker in empty_markers)

    def _collect_evidence(self, question: str, intent: QuestionIntent, owner_scope: str | None = None) -> dict[str, Any]:
        entity_term = self._extract_entity_hint(question)
        evidence: dict[str, Any] = {
            "sources": ["warehouse.db"],
            "entity_hint": entity_term,
            "direct_answer": None,
        }

        with self._connect() as conn:
            if entity_term:
                evidence["matches"] = self._entity_matches(conn, entity_term, owner_scope=owner_scope)
                evidence["recent_interactions"] = self._recent_interactions_for_entity(conn, entity_term, owner_scope=owner_scope)
                evidence["recent_notes"] = self._recent_notes_for_entity(conn, entity_term, owner_scope=owner_scope)
                evidence["contact_rows"] = [row.__dict__ for row in self._contact_rows_for_entity(conn, entity_term, owner_scope=owner_scope)]

            if entity_term and (intent.asks_for_contact_directory or (intent.asks_for_names and intent.asks_for_phones)):
                evidence["direct_answer"] = self._contact_directory_for_entity(conn, entity_term, owner_scope=owner_scope)
            elif entity_term and intent.asks_for_names and intent.asks_for_emails:
                evidence["direct_answer"] = self._contact_points_for_entity(conn, entity_term, owner_scope=owner_scope)
            elif entity_term and intent.asks_for_emails:
                evidence["direct_answer"] = self._emails_for_entity(conn, entity_term, owner_scope=owner_scope)
            elif entity_term and intent.asks_for_phones:
                evidence["direct_answer"] = self._phones_for_entity(conn, entity_term, owner_scope=owner_scope)
            elif intent.asks_for_assigned_clients:
                effective_owner = owner_scope or self._owner_scope_from_question(conn, question)
                evidence["assigned_clients"] = self._assigned_clients(conn, effective_owner)
                evidence["direct_answer"] = self._format_assigned_clients(evidence["assigned_clients"], effective_owner)
            elif intent.asks_for_owner_load:
                evidence["owner_load"] = self._owner_load(conn)
                evidence["direct_answer"] = self._format_owner_load(evidence["owner_load"])
            elif intent.asks_for_interactions_by_owner:
                evidence["interactions_by_owner"] = self._interactions_by_owner(conn, owner_scope)
                evidence["direct_answer"] = self._format_interactions_by_owner(evidence["interactions_by_owner"], owner_scope)
            elif intent.asks_for_generic_relative_interactions:
                days_ago = 2 if "antier" in self._normalize_search_text(question) or "anteayer" in self._normalize_search_text(question) else 1
                evidence["relative_interactions"] = self._interactions_on_relative_day(conn, days_ago, owner_scope)
                fallback = "No encontre interacciones registradas antier." if days_ago == 2 else "No encontre interacciones registradas ayer."
                evidence["direct_answer"] = self._format_interaction_list(evidence["relative_interactions"], fallback)
            elif intent.asks_for_recent_activity_by_owner:
                evidence["recent_activity_by_owner"] = self._recent_activity_by_owner(conn, owner_scope)
                evidence["direct_answer"] = self._format_recent_activity_by_owner(evidence["recent_activity_by_owner"], owner_scope)
            elif intent.asks_for_yesterday_contacts:
                evidence["yesterday_contacts"] = self._interactions_on_relative_day(conn, 1, owner_scope)
                evidence["direct_answer"] = self._format_interaction_list(evidence["yesterday_contacts"], "No encontre contactos registrados ayer.")
            elif intent.asks_for_day_before_yesterday_contacts:
                evidence["day_before_yesterday_contacts"] = self._interactions_on_relative_day(conn, 2, owner_scope)
                evidence["direct_answer"] = self._format_interaction_list(evidence["day_before_yesterday_contacts"], "No encontre contactos registrados antier.")
            elif intent.asks_for_latest_contacted:
                evidence["latest_contacted"] = self._latest_contacted(conn, owner_scope)
                evidence["direct_answer"] = self._format_latest_contacted(evidence["latest_contacted"])
            elif intent.asks_for_today_pending:
                evidence["today_pending"] = self._today_pending(conn, owner_scope)
                evidence["direct_answer"] = self._format_today_pending(evidence["today_pending"])
            elif intent.asks_for_pending_commitments:
                evidence["pending_tasks"] = self._pending_tasks(conn, owner_scope)
                evidence["direct_answer"] = self._format_pending_tasks(evidence["pending_tasks"])
            elif intent.asks_for_stale_contacts:
                evidence["stale_contacts"] = self._stale_contacts(conn, owner_scope)
                evidence["direct_answer"] = self._format_stale_contacts(evidence["stale_contacts"])
            elif intent.asks_for_today_call_list:
                evidence["today_call_list"] = self.get_priority_followups(owner=owner_scope, limit=15)
                evidence["direct_answer"] = (
                    self._format_today_call_list(evidence["today_call_list"])
                    if intent.mode == "data"
                    else self._format_today_call_analysis(evidence["today_call_list"], owner_scope)
                )
            elif intent.asks_for_comparison:
                evidence["comparison_candidates"] = self._comparison_candidates(conn, question, owner_scope)
                evidence["direct_answer"] = (
                    self._format_comparison_candidates(evidence["comparison_candidates"])
                    if intent.mode == "data"
                    else self._format_comparison_analysis(evidence["comparison_candidates"])
                )
            elif intent.asks_for_last_contact:
                evidence["last_interactions"] = self._last_interactions(conn, entity_term, owner_scope)
                evidence["direct_answer"] = self._format_last_interactions(evidence["last_interactions"])
            elif intent.asks_for_kpis:
                window_days = 7 if intent.asks_for_weekly_window else None
                if intent.asks_for_global_kpis or "de todos los vendedores" in self._normalize_search_text(question):
                    evidence["global_kpis"] = self._global_kpis(conn, window_days)
                    evidence["direct_answer"] = self._format_global_kpis(evidence["global_kpis"], window_days)
                else:
                    evidence["owner_kpis"] = self._owner_kpis(conn, owner_scope, window_days)
                    evidence["direct_answer"] = self._format_owner_kpis(evidence["owner_kpis"], owner_scope, window_days)
            elif intent.asks_for_risks and entity_term:
                evidence["risk_profile"] = self._risk_profile(conn, entity_term, owner_scope)
                evidence["direct_answer"] = self._format_risk_profile(evidence["risk_profile"], entity_term)

            evidence["document_chunks"] = self._document_search(conn, question, entity_term)
            if evidence["document_chunks"]:
                evidence["sources"].extend(sorted({chunk["file_name"] for chunk in evidence["document_chunks"]}))

        return evidence

    def _extract_entity_hint(self, question: str) -> str | None:
        normalized = self._normalize_search_text(question)
        patterns = [
            r"(?:correos?|emails?|telefonos?|numeros?|nombres?|contactos?)\s+(?:de|del)\s+(.+)$",
            r"(?:cliente|prospecto|contacto|empresa)\s*:\s*(.+)$",
            r"(?:plan|accion|riesgos?|ultimo contacto)\s+(?:de|del|para)\s+(.+)$",
            r"(?:de|del|para)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                candidate = self._clean_search_term(match.group(1).strip(" ?.:"))
                if candidate:
                    return candidate
        if ":" in normalized:
            trailing = self._clean_search_term(normalized.rsplit(":", 1)[-1].strip(" ?.:"))
            if trailing:
                return trailing
        return None

    def _entity_matches(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> list[dict[str, Any]]:
        wildcard = f"%{self._clean_search_term(term)}%"
        query = """
        SELECT *
        FROM (
            SELECT
                'lead' AS entity_type,
                id,
                owner_name,
                company_name,
                contact_name,
                full_name,
                email,
                phone,
                last_activity_time
            FROM leads
            UNION ALL
            SELECT
                'contact' AS entity_type,
                id,
                owner_name,
                company_name,
                contact_name,
                full_name,
                email,
                phone,
                last_activity_time
            FROM contacts
        )
        WHERE (
            lower(coalesce(company_name, '')) LIKE lower(?)
            OR lower(coalesce(contact_name, '')) LIKE lower(?)
            OR lower(coalesce(full_name, '')) LIKE lower(?)
            OR lower(coalesce(email, '')) LIKE lower(?)
            OR lower(coalesce(phone, '')) LIKE lower(?)
        )
        """
        params: list[Any] = [wildcard, wildcard, wildcard, wildcard, wildcard]
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY last_activity_time DESC, company_name, contact_name LIMIT 25"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _recent_interactions_for_entity(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> list[dict[str, Any]]:
        wildcard = f"%{self._clean_search_term(term)}%"
        query = """
        SELECT related_name, related_module, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE lower(coalesce(related_name, '')) LIKE lower(?)
        """
        params: list[Any] = [wildcard]
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY interaction_at DESC LIMIT 12"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _recent_notes_for_entity(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> list[dict[str, Any]]:
        wildcard = f"%{self._clean_search_term(term)}%"
        query = """
        SELECT parent_name, parent_module, owner_name, created_time, title, content_text
        FROM notes
        WHERE (
            lower(coalesce(parent_name, '')) LIKE lower(?)
            OR lower(coalesce(content_text, '')) LIKE lower(?)
        )
        """
        params: list[Any] = [wildcard, wildcard]
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY created_time DESC LIMIT 12"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _contact_rows_for_entity(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> list[ContactRow]:
        rows: list[ContactRow] = []
        seen: set[str] = set()

        for match in self._entity_matches(conn, term, owner_scope=owner_scope):
            label = match.get("contact_name") or match.get("full_name") or match.get("company_name") or "Sin nombre"
            email = match.get("email") or "Sin correo"
            phone = match.get("phone") or "Sin telefono"
            row = ContactRow(
                label=label,
                company=match.get("company_name") or "",
                email=email,
                phone=phone,
                owner_name=match.get("owner_name") or "Sin responsable",
                source=match.get("entity_type") or "crm",
            )
            key = self._contact_row_key(row)
            if key not in seen:
                seen.add(key)
                rows.append(row)

        for note in self._recent_notes_for_entity(conn, term, owner_scope=owner_scope):
            for row in self._extract_contacts_from_note(note):
                key = self._contact_row_key(row)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

        rows.sort(
            key=lambda item: (
                self._sort_sentinel(item.label == "Sin nombre"),
                self._normalize_search_text(item.label),
                self._normalize_search_text(item.company),
                item.email.lower(),
            )
        )
        return rows

    def _extract_contacts_from_note(self, note: dict[str, Any]) -> list[ContactRow]:
        content = strip_html(note.get("content_text") or note.get("content_raw") or "")
        if not content.strip():
            return []

        extracted: list[ContactRow] = []
        emails = [self._clean_email(match.group(0)) for match in EMAIL_RE.finditer(content)]
        phones = [self._clean_phone(match.group(0)) for match in PHONE_RE.finditer(content)]

        email_phone_pairs = list(zip(emails, phones))
        assigned_phones = {phone for _, phone in email_phone_pairs}

        for email in emails:
            name = self._guess_contact_name_from_note(content, email) or "Contacto en nota"
            phone = self._phone_near_email(content, email) or "Sin telefono"
            extracted.append(
                ContactRow(
                    label=name,
                    company=note.get("parent_name") or "",
                    email=email,
                    phone=phone,
                    owner_name=note.get("owner_name") or "Sin responsable",
                    source="nota CRM",
                )
            )

        for phone in phones:
            if phone in assigned_phones:
                continue
            name = self._guess_contact_name_for_phone(content, phone) or "Contacto en nota"
            extracted.append(
                ContactRow(
                    label=name,
                    company=note.get("parent_name") or "",
                    email="Sin correo",
                    phone=phone,
                    owner_name=note.get("owner_name") or "Sin responsable",
                    source="nota CRM",
                )
            )
        return extracted

    def _emails_for_entity(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> str:
        rows = self._contact_rows_for_entity(conn, term, owner_scope=owner_scope)
        emails = sorted({row.email for row in rows if row.email and row.email != "Sin correo"})
        return "\n".join(emails) if emails else "No encontre correos para ese cliente o prospecto."

    def _phones_for_entity(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> str:
        rows = self._contact_rows_for_entity(conn, term, owner_scope=owner_scope)
        phones = []
        for row in rows:
            if not row.phone or row.phone == "Sin telefono":
                continue
            phones.append(f"{row.label} | {row.phone}")
        unique = list(dict.fromkeys(phones))
        return "\n".join(unique) if unique else "No encontre telefonos para ese cliente o prospecto."

    def _contact_points_for_entity(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> str:
        rows = self._contact_rows_for_entity(conn, term, owner_scope=owner_scope)
        lines = []
        seen: set[str] = set()
        for row in rows:
            if not row.email or row.email == "Sin correo":
                continue
            key = f"{self._normalize_search_text(row.label)}|{row.email.lower()}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {row.label} | {row.email}")
        return "\n".join(lines) if lines else "No encontre nombres y correos para ese cliente o prospecto."

    def _contact_directory_for_entity(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> str:
        rows = self._contact_rows_for_entity(conn, term, owner_scope=owner_scope)
        if not rows:
            return f"No hay evidencia disponible sobre contactos de {term} en la informacion proporcionada."
        lines = []
        for row in rows:
            suffix = f" | Empresa: {row.company}" if row.company else ""
            source = f" | Fuente: {row.source}" if row.source == "nota CRM" else ""
            lines.append(
                f"- {row.label} | Correo: {row.email} | Telefono: {row.phone} | Responsable CRM: {row.owner_name}{suffix}{source}"
            )
        return f"Contactos de {term} disponibles:\n\n" + "\n".join(lines)

    def _owner_load(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, COUNT(*) AS total_records
        FROM (
            SELECT owner_name FROM leads
            UNION ALL
            SELECT owner_name FROM contacts
        )
        WHERE owner_name IS NOT NULL AND trim(owner_name) <> ''
        GROUP BY owner_name
        ORDER BY total_records DESC, owner_name ASC
        """
        return [dict(row) for row in conn.execute(query).fetchall()]

    def _assigned_clients(self, conn: sqlite3.Connection, owner_scope: str | None) -> list[dict[str, Any]]:
        if not owner_scope:
            return []
        query = """
        SELECT
            entity_type,
            company_name,
            contact_name,
            full_name,
            email,
            phone,
            owner_name,
            last_activity_time
        FROM (
            SELECT
                'lead' AS entity_type,
                company_name,
                contact_name,
                full_name,
                email,
                phone,
                owner_name,
                last_activity_time
            FROM leads
            UNION ALL
            SELECT
                'contact' AS entity_type,
                company_name,
                contact_name,
                full_name,
                email,
                phone,
                owner_name,
                last_activity_time
            FROM contacts
        )
        WHERE owner_name = ?
        ORDER BY last_activity_time DESC, company_name, contact_name
        LIMIT 50
        """
        return [dict(row) for row in conn.execute(query, (owner_scope,)).fetchall()]

    def _interactions_by_owner(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, COUNT(*) AS total_interactions
        FROM interactions
        WHERE owner_name IS NOT NULL AND trim(owner_name) <> ''
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " GROUP BY owner_name ORDER BY total_interactions DESC, owner_name ASC"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _recent_activity_by_owner(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, related_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE owner_name IS NOT NULL
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY interaction_at DESC LIMIT 20"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _interactions_on_relative_day(self, conn: sqlite3.Connection, days_ago: int, owner_scope: str | None = None) -> list[dict[str, Any]]:
        target_date = datetime.now().date() - timedelta(days=days_ago)
        query = """
        SELECT related_name, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE date(interaction_at) = date(?)
        """
        params: list[Any] = [target_date.isoformat()]
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY interaction_at DESC LIMIT 20"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _latest_contacted(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> dict[str, Any] | None:
        query = """
        SELECT related_name, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE related_name IS NOT NULL
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY interaction_at DESC LIMIT 1"
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    def _pending_tasks(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, contact_name, subject, status, priority, due_date
        FROM tasks
        WHERE status IS NULL OR lower(status) NOT IN ('completed', 'cancelled')
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY due_date IS NULL, due_date ASC, owner_name ASC LIMIT 20"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _today_pending(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, contact_name, subject, status, priority, due_date
        FROM tasks
        WHERE date(due_date) = date('now')
          AND (status IS NULL OR lower(status) NOT IN ('completed', 'cancelled'))
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY due_date ASC, owner_name ASC LIMIT 20"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _stale_contacts(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        WITH latest AS (
            SELECT related_id, related_name, owner_name, MAX(interaction_at) AS last_touch
            FROM interactions
            WHERE related_id IS NOT NULL
            GROUP BY related_id, related_name, owner_name
        )
        SELECT related_name, owner_name, last_touch
        FROM latest
        WHERE julianday('now') - julianday(last_touch) >= 30
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY last_touch ASC LIMIT 20"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _last_interactions(self, conn: sqlite3.Connection, term: str | None, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT related_name, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE 1 = 1
        """
        params: list[Any] = []
        if term:
            query += " AND lower(coalesce(related_name, '')) LIKE lower(?)"
            params.append(f"%{self._clean_search_term(term)}%")
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY interaction_at DESC LIMIT 10"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _owner_kpis(self, conn: sqlite3.Connection, owner_scope: str | None, window_days: int | None = None) -> dict[str, Any]:
        if not owner_scope:
            return {}
        where_clause, params = self._interaction_window_clause(owner_scope, window_days)
        totals = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_interactions,
                COUNT(DISTINCT related_id) AS unique_accounts
            FROM interactions
            {where_clause}
            """,
            params,
        ).fetchone()
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
        return {
            "owner_name": owner_scope,
            "window_days": window_days,
            "total_interactions": totals["total_interactions"] if totals else 0,
            "unique_accounts": totals["unique_accounts"] if totals else 0,
            "by_type": by_type,
        }

    def _global_kpis(self, conn: sqlite3.Connection, window_days: int | None = None) -> dict[str, Any]:
        where_clause, params = self._interaction_window_clause(None, window_days)
        totals = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_interactions,
                COUNT(DISTINCT owner_name) AS owners,
                COUNT(DISTINCT related_id) AS unique_accounts
            FROM interactions
            {where_clause}
            """,
            params,
        ).fetchone()
        by_owner = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT owner_name, COUNT(*) AS total
                FROM interactions
                {where_clause}
                GROUP BY owner_name
                ORDER BY total DESC
                LIMIT 12
                """,
                params,
            ).fetchall()
        ]
        return {
            "window_days": window_days,
            "total_interactions": totals["total_interactions"] if totals else 0,
            "owners": totals["owners"] if totals else 0,
            "unique_accounts": totals["unique_accounts"] if totals else 0,
            "by_owner": by_owner,
        }

    def _risk_profile(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> dict[str, Any]:
        matches = self._entity_matches(conn, term, owner_scope=owner_scope)
        notes = self._recent_notes_for_entity(conn, term, owner_scope=owner_scope)
        interactions = self._recent_interactions_for_entity(conn, term, owner_scope=owner_scope)
        combined = " || ".join((note.get("content_text") or "") for note in notes).lower()
        risks: list[str] = []
        if "rechaz" in combined or "no cree" in combined:
            risks.append("Hay objeciones o rechazo en notas recientes.")
        if "rebot" in combined:
            risks.append("Hay correos que rebotaron, el dato de contacto puede estar desactualizado.")
        if not interactions:
            risks.append("No hay interacciones recientes registradas.")
        if not matches:
            risks.append("No hay registro estructurado visible en leads o contacts.")
        return {
            "entity": term,
            "match_count": len(matches),
            "interaction_count": len(interactions),
            "note_count": len(notes),
            "risks": risks,
        }

    def _comparison_candidates(self, conn: sqlite3.Connection, question: str, owner_scope: str | None = None) -> list[dict[str, Any]]:
        normalized = self._normalize_search_text(question)
        candidates: list[str] = []
        if " vs " in normalized:
            candidates = [self._clean_search_term(part) for part in normalized.split(" vs ", 1)]
        else:
            match = re.search(r"compara(?:r)?\s+(.+?)\s+(?:con|vs)\s+(.+)$", normalized)
            if match:
                candidates = [self._clean_search_term(match.group(1)), self._clean_search_term(match.group(2))]
        cleaned = [candidate for candidate in candidates if candidate]
        results = []
        for candidate in cleaned[:2]:
            matches = self._entity_matches(conn, candidate, owner_scope=owner_scope)
            interactions = self._recent_interactions_for_entity(conn, candidate, owner_scope=owner_scope)
            notes = self._recent_notes_for_entity(conn, candidate, owner_scope=owner_scope)
            results.append(
                {
                    "entity": candidate,
                    "matches": len(matches),
                    "recent_interactions": len(interactions),
                    "recent_notes": len(notes),
                    "owners": sorted({row.get("owner_name") for row in matches if row.get("owner_name")}),
                    "score": self._comparison_score(matches, interactions, notes),
                }
            )
        return results

    def _document_search(self, conn: sqlite3.Connection, question: str, entity_term: str | None = None) -> list[dict[str, Any]]:
        terms = []
        normalized_question = self._normalize_search_text(question)
        if entity_term:
            terms.append(entity_term)
        terms.extend([term for term in normalized_question.split() if len(term) >= 4])
        unique_terms = []
        seen: set[str] = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            unique_terms.append(term)
        unique_terms = unique_terms[:6]
        if not unique_terms:
            return []

        clauses = " OR ".join("lower(content) LIKE lower(?)" for _ in unique_terms)
        query = f"""
        SELECT d.file_name, dc.chunk_index, dc.content
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE {clauses}
        LIMIT 8
        """
        values = [f"%{term}%" for term in unique_terms]
        try:
            return [dict(row) for row in conn.execute(query, values).fetchall()]
        except sqlite3.OperationalError:
            return []

    def _mentioned_owner_names(self, question: str) -> list[str]:
        normalized = self._normalize_search_text(question)
        aliases: dict[str, str] = {
            "eduardo": "Eduardo Valdez",
            "evaldez": "Eduardo Valdez",
            "eduardo valdez": "Eduardo Valdez",
            "emmanuel": "Jesus Emmanuel Meza Guzmán",
            "emeza": "Jesus Emmanuel Meza Guzmán",
            "jesus emmanuel meza": "Jesus Emmanuel Meza Guzmán",
            "jesus emmanuel meza guzman": "Jesus Emmanuel Meza Guzmán",
            "pablo": "Pablo Melin Dorador",
            "pmelin": "Pablo Melin Dorador",
            "pablo melin": "Pablo Melin Dorador",
        }
        for user in APP_USERS.values():
            if user.crm_owner_name:
                key = self._normalize_search_text(user.crm_owner_name)
                aliases[key] = user.crm_owner_name

        with self._connect() as conn:
            for row in conn.execute("SELECT DISTINCT name FROM owners WHERE name IS NOT NULL").fetchall():
                owner_name = row[0]
                key = self._normalize_search_text(owner_name)
                aliases[key] = owner_name

        matches = []
        for alias, owner_name in aliases.items():
            if alias and alias in normalized:
                matches.append(owner_name)
        return list(dict.fromkeys(matches))

    def _resolve_owner_scope(self, question: str, user: AppUser | None) -> str | None:
        normalized = self._normalize_search_text(question)
        global_markers = [
            "global",
            "todos los vendedores",
            "todo el equipo",
            "equipo comercial",
            "todos",
        ]
        if any(marker in normalized for marker in global_markers) and any(word in normalized for word in ["vendedores", "agentes", "equipo", "kpi", "kpis", "clientes"]):
            return None

        mentioned = self._mentioned_owner_names(question)
        if mentioned:
            return mentioned[0]

        if user:
            padded = f" {normalized} "
            self_scope_signals = [
                " mi ",
                " mis ",
                " conmigo ",
                " yo ",
                " debo ",
                " tengo ",
                " hable ",
                " llame ",
                " mis clientes",
                " mis prospectos",
                " mis notas",
                " mi cartera",
            ]
            if any(signal in padded for signal in self_scope_signals):
                return user.crm_owner_name

        if not user or user.role != "seller":
            return None

        seller_default_signals = [
            "a quien debo llamar hoy",
            "a quien le hable ayer",
            "a quien le hable antier",
            "compromisos pendientes para hoy",
            "interacciones de ayer",
            "interacciones de antier",
            "actividad reciente",
            "mis clientes",
            "mis prospectos",
            "mi cartera",
        ]
        if any(self._normalize_search_text(signal) in normalized for signal in seller_default_signals):
            return user.crm_owner_name
        return None

    def _normalize_user_scoped_question(self, question: str, owner_scope: str | None) -> str:
        if not owner_scope:
            return question
        normalized = self._normalize_search_text(question)
        if any(token in f" {normalized} " for token in [" mi ", " mis ", " yo ", " conmigo "]):
            return f"{question.strip()} del vendedor {owner_scope}"
        return question

    def _owner_scope_from_question(self, conn: sqlite3.Connection, question: str) -> str | None:
        mentioned = self._mentioned_owner_names(question)
        if mentioned:
            return mentioned[0]
        normalized = self._normalize_search_text(question)
        rows = conn.execute("SELECT DISTINCT name FROM owners WHERE name IS NOT NULL").fetchall()
        for row in rows:
            owner_name = row[0]
            if self._normalize_search_text(owner_name) in normalized:
                return owner_name
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

    def _normalize_search_text(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
        return " ".join(ascii_text.lower().split())

    def _clean_search_term(self, value: str) -> str:
        cleaned = self._normalize_search_text(value or "")
        removable_prefixes = [
            "compara ",
            "comparar ",
            "comparativa ",
            "dame ",
            "solo ",
            "cliente ",
            "prospecto ",
            "contacto ",
            "empresa ",
        ]
        changed = True
        while changed:
            changed = False
            for prefix in removable_prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    changed = True
        return cleaned.strip()

    def _clean_email(self, value: str) -> str:
        return value.strip().strip(".,;:()[]<>").lower()

    def _clean_phone(self, value: str) -> str:
        cleaned = re.sub(r"[^\d+]", "", value)
        if len(cleaned) < 7:
            return value.strip()
        return cleaned

    def _phone_near_email(self, content: str, email: str) -> str | None:
        parts = content.split(email)
        if len(parts) < 2:
            return None
        window = (parts[0][-120:] + " " + parts[1][:120]).strip()
        match = PHONE_RE.search(window)
        if not match:
            return None
        return self._clean_phone(match.group(0))

    def _guess_contact_name_from_note(self, content: str, email: str) -> str | None:
        lowered = content.lower()
        index = lowered.find(email.lower())
        local_name = email.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
        if index >= 0:
            pre_context = content[max(0, index - 120):index]
            post_context = content[index:min(len(content), index + len(email) + 120)]
            if "en copia" in self._normalize_search_text(post_context) and local_name:
                return " ".join(part.capitalize() for part in local_name.split())
        window = content
        if index >= 0:
            start = max(0, index - 180)
            end = min(len(content), index + len(email) + 180)
            window = content[start:end]
        patterns = [
            r"contacto correcto de\s+([A-Za-zÁÉÍÓÚÑáéíóúñ ]+)\s*\([^)]*\)\s*$",
            r"correo de\s+([A-Za-zÁÉÍÓÚÑáéíóúñ ]+)\s*$",
            r"preguntando por\s+([A-Za-zÁÉÍÓÚÑáéíóúñ ]+)\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, window[: window.lower().find(email.lower())] if email.lower() in window.lower() else window, flags=re.IGNORECASE)
            if match:
                candidate = self._clean_contact_name_guess(match.group(1))
                if candidate:
                    return candidate
        if local_name:
            return " ".join(part.capitalize() for part in local_name.split())
        return None

    def _guess_contact_name_for_phone(self, content: str, phone: str) -> str | None:
        escaped = re.escape(phone)
        patterns = [
            rf"([A-Za-zÁÉÍÓÚÑáéíóúñ ]+)\s*(?:telefono|tel|cel|movil|whatsapp)?\s*[:\-]?\s*{escaped}",
            rf"contacto\s+([A-Za-zÁÉÍÓÚÑáéíóúñ ]+).*?{escaped}",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
            if match:
                candidate = self._clean_contact_name_guess(match.group(1))
                if candidate:
                    return candidate
        return None

    def _clean_contact_name_guess(self, value: str | None) -> str | None:
        if not value:
            return None
        candidate = " ".join(str(value).split()).strip(" ,.;:-")
        normalized = self._normalize_search_text(candidate)
        banned_terms = {
            "correo",
            "encargada",
            "encargado",
            "encargados",
            "monitoreo",
            "combustible",
            "contacto",
            "gps",
            "recepcion",
            "llamada",
        }
        words = normalized.split()
        if len(words) > 4:
            return None
        if any(word in banned_terms for word in words):
            return None
        return candidate or None

    def _contact_row_key(self, row: ContactRow) -> str:
        return "|".join(
            [
                self._normalize_search_text(row.label),
                row.email.lower(),
                row.phone,
                self._normalize_search_text(row.company),
            ]
        )

    def _sort_sentinel(self, value: bool) -> int:
        return 1 if value else 0

    def _priority_label(self, score: int) -> str:
        if score >= 50:
            return "alta"
        if score >= 35:
            return "media"
        return "normal"

    def _comparison_score(self, matches: list[dict[str, Any]], interactions: list[dict[str, Any]], notes: list[dict[str, Any]]) -> int:
        score = 0
        if matches:
            score += 10
        if any(match.get("owner_name") for match in matches):
            score += 10
        score += min(20, len(interactions) * 4)
        score += min(20, len(notes) * 3)
        combined = " || ".join((note.get("content_text") or "").lower() for note in notes)
        if "rechaz" in combined or "no cree" in combined:
            score -= 8
        if "espera de respuesta" in combined or "pendiente" in combined:
            score += 4
        return score

    def _interaction_window_clause(self, owner_scope: str | None, window_days: int | None) -> tuple[str, list[Any]]:
        filters = ["owner_name IS NOT NULL"]
        params: list[Any] = []
        if owner_scope:
            filters.append("owner_name = ?")
            params.append(owner_scope)
        if window_days:
            filters.append("date(interaction_at) >= date(?)")
            params.append((datetime.now().date() - timedelta(days=window_days)).isoformat())
        return f"WHERE {' AND '.join(filters)}", params

    def _format_owner_load(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre propietarios con registros."
        return "\n".join(f"{row['owner_name']}: {row['total_records']}" for row in rows)

    def _format_assigned_clients(self, rows: list[dict[str, Any]], owner_scope: str | None) -> str:
        if not rows:
            return f"No encontre clientes o prospectos asignados a {owner_scope}." if owner_scope else "No encontre clientes asignados."
        lines = []
        for row in rows:
            label = row.get("company_name") or row.get("contact_name") or row.get("full_name") or "Sin nombre"
            email = row.get("email") or "Sin correo"
            phone = row.get("phone") or "Sin telefono"
            lines.append(f"- {label} | {email} | {phone}")
        owner_label = owner_scope or "ese vendedor"
        return f"Clientes y prospectos de {owner_label}:\n\n" + "\n".join(lines)

    def _format_interactions_by_owner(self, rows: list[dict[str, Any]], owner_scope: str | None) -> str:
        if not rows:
            return "No encontre interacciones por vendedor."
        if owner_scope and len(rows) == 1:
            row = rows[0]
            return f"{row['owner_name']}: {row['total_interactions']} interacciones"
        return "\n".join(f"{row['owner_name']}: {row['total_interactions']}" for row in rows)

    def _format_recent_activity_by_owner(self, rows: list[dict[str, Any]], owner_scope: str | None) -> str:
        if not rows:
            return "No encontre actividad reciente."
        return "\n".join(
            f"{row.get('owner_name') or 'Sin responsable'} | {row.get('related_name') or 'Sin nombre'} | {row.get('source_type') or '-'} | {row.get('interaction_at') or '-'}"
            for row in rows
        )

    def _format_pending_tasks(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre compromisos pendientes."
        return "\n".join(
            f"{row.get('owner_name') or 'Sin responsable'} | {row.get('contact_name') or 'Sin contacto'} | {row.get('subject') or 'Sin asunto'} | {row.get('status') or 'Sin estatus'} | {row.get('due_date') or 'Sin fecha'}"
            for row in rows
        )

    def _format_today_pending(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre compromisos pendientes para hoy."
        return "\n".join(
            f"{row.get('owner_name') or 'Sin responsable'} | {row.get('contact_name') or 'Sin contacto'} | {row.get('subject') or 'Sin asunto'} | {row.get('priority') or 'Sin prioridad'} | {row.get('due_date') or 'Sin fecha'}"
            for row in rows
        )

    def _format_stale_contacts(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre clientes con mas de 30 dias sin contacto."
        return "\n".join(
            f"{row.get('related_name') or 'Sin nombre'} | {row.get('owner_name') or 'Sin responsable'} | {row.get('last_touch') or '-'}"
            for row in rows
        )

    def _format_interaction_list(self, rows: list[dict[str, Any]], fallback: str) -> str:
        if not rows:
            return fallback
        return "\n".join(
            f"{row.get('related_name') or 'Sin nombre'} | {row.get('source_type') or '-'} | {row.get('interaction_at') or '-'} | {row.get('owner_name') or 'Sin responsable'}"
            for row in rows
        )

    def _format_latest_contacted(self, row: dict[str, Any] | None) -> str:
        if not row:
            return "No encontre un ultimo contacto registrado."
        return (
            f"{row.get('related_name') or 'Sin nombre'} | "
            f"{row.get('source_type') or '-'} | "
            f"{row.get('interaction_at') or '-'} | "
            f"{row.get('owner_name') or 'Sin responsable'}"
        )

    def _format_today_call_list(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre clientes sugeridos para llamar hoy."
        return "\n".join(
            f"{row.get('related_name') or 'Sin nombre'} | {row.get('owner_name') or 'Sin responsable'} | prioridad {row.get('priority_label') or '-'} | score {row.get('score') or 0} | {' / '.join(row.get('reasons') or [])}"
            for row in rows
        )

    def _format_today_call_analysis(self, rows: list[dict[str, Any]], owner_scope: str | None) -> str:
        if not rows:
            owner_label = owner_scope or "el equipo"
            return (
                "Hechos:\n"
                f"- No se detectaron clientes priorizados para llamar hoy para {owner_label}.\n\n"
                "Interpretacion:\n"
                "- No hay evidencia suficiente de seguimiento vencido o compromisos abiertos para priorizar una llamada hoy.\n\n"
                "Recomendacion:\n"
                "- Revisa notas recientes, nuevas tareas o sincroniza Zoho para confirmar si entraron pendientes nuevos."
            )
        top = rows[:5]
        facts = [f"- {row['related_name']} aparece con prioridad {row['priority_label']} y score {row['score']}." for row in top]
        reasons = []
        for row in top[:3]:
            joined = " ".join(row.get("reasons") or [])
            reasons.append(f"- {row['related_name']}: {joined}")
        recommendations = [f"- Prioriza primero a {row['related_name']}." for row in top[:3]]
        return "\n".join(
            [
                "Hechos:",
                *facts,
                "",
                "Interpretacion:",
                *reasons,
                "",
                "Recomendacion:",
                *recommendations,
            ]
        )

    def _format_comparison_candidates(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre evidencia suficiente para comparar."
        return "\n".join(
            f"{row['entity']}: score {row['score']} | coincidencias {row['matches']} | interacciones {row['recent_interactions']} | notas {row['recent_notes']}"
            for row in rows
        )

    def _format_comparison_analysis(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return (
                "Hechos:\n"
                "- No encontre evidencia suficiente para comparar las entidades solicitadas.\n\n"
                "Interpretacion:\n"
                "- La base no ofrece señales suficientes para priorizar una sobre otra.\n\n"
                "Recomendacion:\n"
                "- Registra actividad o agrega informacion comercial antes de compararlas."
            )
        facts = [
            f"- {row['entity']}: {row['matches']} coincidencias, {row['recent_interactions']} interacciones y {row['recent_notes']} notas."
            for row in rows
        ]
        best = max(rows, key=lambda item: item["score"])
        return "\n".join(
            [
                "Hechos:",
                *facts,
                "",
                "Interpretacion:",
                f"- {best['entity']} tiene la evidencia comercial mas fuerte dentro del CRM actual.",
                "",
                "Recomendacion:",
                f"- Prioriza {best['entity']} si necesitas decidir con la informacion disponible hoy.",
            ]
        )

    def _format_last_interactions(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre interacciones recientes."
        return "\n".join(
            f"{row.get('related_name') or 'Sin nombre'} | {row.get('source_type') or '-'} | {row.get('interaction_at') or '-'} | {row.get('owner_name') or 'Sin responsable'}"
            for row in rows
        )

    def _format_owner_kpis(self, kpis: dict[str, Any], owner_scope: str | None, window_days: int | None) -> str:
        if not kpis:
            return "No encontre KPIs para el vendedor solicitado."
        scope = owner_scope or kpis.get("owner_name") or "el vendedor"
        header = f"KPIs de {scope}"
        if window_days:
            header += f" en los ultimos {window_days} dias"
        lines = [
            header,
            f"Interacciones: {kpis.get('total_interactions', 0)}",
            f"Cuentas unicas: {kpis.get('unique_accounts', 0)}",
        ]
        for row in kpis.get("by_type", []):
            lines.append(f"{row['source_type']}: {row['total']}")
        return "\n".join(lines)

    def _format_global_kpis(self, kpis: dict[str, Any], window_days: int | None) -> str:
        if not kpis:
            return "No encontre KPIs globales para el periodo solicitado."
        header = "KPIs globales"
        if window_days:
            header += f" en los ultimos {window_days} dias"
        lines = [
            header,
            f"Interacciones: {kpis.get('total_interactions', 0)}",
            f"Vendedores con actividad: {kpis.get('owners', 0)}",
            f"Cuentas unicas: {kpis.get('unique_accounts', 0)}",
        ]
        for row in kpis.get("by_owner", []):
            lines.append(f"{row['owner_name']}: {row['total']}")
        return "\n".join(lines)

    def _format_risk_profile(self, profile: dict[str, Any], entity_term: str) -> str:
        risks = profile.get("risks") or []
        if not risks:
            return f"No detecte riesgos claros para {entity_term} con la evidencia actual."
        lines = [f"Riesgos detectados para {entity_term}:"]
        lines.extend(f"- {risk}" for risk in risks)
        return "\n".join(lines)

    def _format_evidence_fallback(self, evidence: dict[str, Any]) -> str:
        sections: list[str] = []
        direct_answer = evidence.get("direct_answer")
        if direct_answer:
            sections.append(direct_answer)
        if evidence.get("recent_interactions"):
            sections.append("Interacciones recientes:\n" + self._format_interaction_list(evidence["recent_interactions"], ""))
        if evidence.get("recent_notes"):
            note_lines = []
            for note in evidence["recent_notes"][:5]:
                note_lines.append(f"- {note.get('parent_name') or 'Sin nombre'} | {note.get('created_time') or '-'} | {(note.get('content_text') or '')[:220]}")
            sections.append("Notas CRM:\n" + "\n".join(note_lines))
        if evidence.get("document_chunks"):
            doc_lines = []
            for chunk in evidence["document_chunks"][:4]:
                doc_lines.append(f"- {chunk['file_name']} | chunk {chunk['chunk_index']} | {chunk['content'][:220]}")
            sections.append("Documentos:\n" + "\n".join(doc_lines))
        return "\n\n".join(section for section in sections if section.strip()) or "No encontre evidencia suficiente para responder."

    def _build_prompt(self, question: str, intent: QuestionIntent, evidence: dict[str, Any]) -> str:
        def serialize_rows(rows: list[dict[str, Any]], limit: int = 8) -> str:
            return "\n".join(f"- {row}" for row in rows[:limit]) if rows else "- Sin registros"

        instructions = [
            "Eres el asistente comercial final del proyecto Director Comercial IA.",
            "Responde en español claro y ejecutivo.",
            "Usa solo la evidencia proporcionada de Zoho CRM y PDFs locales.",
            "No inventes contactos, teléfonos, correos, responsables ni conclusiones.",
            "Si hay duda o falta de evidencia, dilo explícitamente.",
        ]
        if intent.mode == "data":
            instructions.append("Como la consulta es de datos, responde de forma breve y directa.")
        else:
            instructions.append("Como la consulta es analítica, separa Hechos, Interpretación y Recomendación cuando ayude.")

        prompt_parts = [
            "\n".join(instructions),
            f"Pregunta: {question}",
            f"Usuario activo: {evidence.get('active_user')}",
            f"Vendedor aplicado: {evidence.get('owner_scope')}",
            f"Entidad detectada: {evidence.get('entity_hint')}",
            "Coincidencias CRM:\n" + serialize_rows(evidence.get("matches", [])),
            "Contactos detectados:\n" + serialize_rows(evidence.get("contact_rows", [])),
            "Interacciones recientes:\n" + serialize_rows(evidence.get("recent_interactions", [])),
            "Notas recientes:\n" + serialize_rows(evidence.get("recent_notes", [])),
            "Clientes asignados:\n" + serialize_rows(evidence.get("assigned_clients", [])),
            "Compromisos pendientes:\n" + serialize_rows(evidence.get("pending_tasks", [])),
            "Prioridades sugeridas:\n" + serialize_rows(evidence.get("today_call_list", [])),
            "Fragmentos de documentos:\n" + serialize_rows(evidence.get("document_chunks", [])),
        ]
        direct_answer = evidence.get("direct_answer")
        if direct_answer:
            prompt_parts.append(f"Respuesta directa sugerida por reglas: {direct_answer}")
        return "\n\n".join(prompt_parts)
