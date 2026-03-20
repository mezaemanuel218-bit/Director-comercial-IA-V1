import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from assistant_core.auth import AppUser
from assistant_core.config import RAW_MODULE_FILES, WAREHOUSE_DB
from assistant_core.query_intent import QuestionIntent, classify_question
from assistant_core.utils import load_json, strip_html


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
        owner_scope = self._resolve_owner_scope_v3(question, user)
        effective_question = self._normalize_user_scoped_question_v3(question, owner_scope)
        evidence = self._collect_evidence(effective_question, intent, owner_scope=owner_scope)
        evidence["active_user"] = self._user_context(user)
        evidence["owner_scope"] = owner_scope

        if evidence.get("direct_answer") and intent.mode == "data":
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

            if entity_term and (
                intent.asks_for_contact_directory
                or (intent.asks_for_names and intent.asks_for_phones)
            ):
                evidence["direct_answer"] = self._contact_directory_for_entity(conn, entity_term)
            elif entity_term and intent.asks_for_names and intent.asks_for_emails:
                evidence["direct_answer"] = self._contact_points_for_entity(conn, entity_term)
            elif intent.asks_for_emails and entity_term:
                evidence["direct_answer"] = self._emails_for_entity(conn, entity_term)
            elif intent.asks_for_assigned_clients:
                effective_owner = owner_scope or self._owner_scope_from_question(question)
                evidence["assigned_clients"] = self._assigned_clients(conn, effective_owner)
                evidence["direct_answer"] = self._format_assigned_clients(evidence["assigned_clients"], effective_owner)
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
            elif intent.asks_for_generic_relative_interactions:
                if "antier" in question.lower() or "anteayer" in question.lower():
                    evidence["relative_interactions"] = self._interactions_on_relative_day(conn, 2, owner_scope)
                    evidence["direct_answer"] = self._format_interaction_list(
                        evidence["relative_interactions"],
                        "No encontre interacciones registradas antier.",
                    )
                else:
                    evidence["relative_interactions"] = self._interactions_on_relative_day(conn, 1, owner_scope)
                    evidence["direct_answer"] = self._format_interaction_list(
                        evidence["relative_interactions"],
                        "No encontre interacciones registradas ayer.",
                    )
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
                owner_comparison = self._owner_comparison(conn, question)
                if owner_comparison:
                    evidence["owner_comparison"] = owner_comparison
                    evidence["direct_answer"] = self._format_owner_comparison(owner_comparison)
                else:
                    evidence["comparison_candidates"] = self._comparison_candidates(conn, question)
                    evidence["direct_answer"] = self._format_comparison_candidates(evidence["comparison_candidates"])
            elif intent.asks_for_last_contact:
                evidence["last_interactions"] = self._last_interactions(conn, entity_term)
                if intent.mode == "data":
                    evidence["direct_answer"] = self._format_last_interactions(evidence["last_interactions"])
            elif intent.asks_for_kpis:
                window_days = 7 if intent.asks_for_weekly_window else None
                if intent.asks_for_global_kpis or "de todos los vendedores" in question.lower():
                    evidence["global_kpis"] = self._global_kpis(conn, window_days=window_days)
                    evidence["direct_answer"] = self._format_global_kpis(evidence["global_kpis"], window_days=window_days)
                else:
                    evidence["owner_kpis"] = self._owner_kpis(conn, owner_scope, window_days=window_days)
                    evidence["direct_answer"] = self._format_owner_kpis(evidence["owner_kpis"], owner_scope, window_days=window_days)
            elif intent.asks_for_risks and entity_term:
                evidence["risk_profile"] = self._risk_profile(conn, entity_term)
                evidence["direct_answer"] = self._format_risk_profile(evidence["risk_profile"], entity_term)

            evidence["document_chunks"] = self._document_search(conn, question)

        return evidence

    def _extract_entity_hint(self, question: str) -> str | None:
        patterns = [
            r"(?:correos?|emails?|telefonos?|teléfonos?|numeros?|números?|nombres?|contactos?)\s+(?:de|del)\s+(.+)$",
            r"(?:cliente|prospecto|contacto|empresa)\s*:\s*(.+)$",
            r"(?:plan|accion|acción|riesgos?|ultimo contacto|último contacto)\s+(?:de|del|para)\s+(.+)$",
            r"(?:de|del|para)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                candidate = self._clean_search_term(match.group(1).strip(" ?.:"))
                if candidate:
                    return candidate
        if ":" in question:
            trailing = self._clean_search_term(question.rsplit(":", 1)[-1].strip(" ?.:"))
            if trailing:
                return trailing
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

    def _resolve_owner_scope_v3(self, question: str, user: AppUser | None) -> str | None:
        normalized = self._normalize_search_text(question)
        global_markers = [
            "global",
            "todos los vendedores",
            "todo el equipo",
            "todo el departamento",
            "equipo comercial",
        ]
        if any(marker in normalized for marker in global_markers):
            return None
        if "todos" in normalized and any(marker in normalized for marker in ["vendedores", "agentes", "propietarios", "equipo", "kpi", "kpis"]):
            return None

        mentioned_owners = self._mentioned_owner_names(question)
        if mentioned_owners:
            return mentioned_owners[0]

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
            " mis interacciones",
            " mi cartera",
        ]
        if user and any(signal in padded for signal in self_scope_signals):
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
            "mis notas",
            "mi cartera",
        ]
        if any(self._normalize_search_text(signal) in normalized for signal in seller_default_signals):
            return user.crm_owner_name
        return None

    def _normalize_user_scoped_question_v3(self, question: str, owner_scope: str | None) -> str:
        if not owner_scope:
            return question
        normalized = self._normalize_search_text(question)
        if any(token in f" {normalized} " for token in [" mi ", " mis ", " yo ", " conmigo "]):
            return f"{question.strip()} del vendedor {owner_scope}"
        return question

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

    def _resolve_owner_scope_v2(self, question: str, user: AppUser | None) -> str | None:
        normalized = self._normalize_search_text(question)
        if any(token in normalized for token in ["global", "todos los vendedores", "todos", "equipo"]):
            return None
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
                "a quien le hable ayer",
                "a quien le hable antier",
                "compromisos pendientes para hoy",
                "kpi",
                "kpis",
            ]
            if any(self._normalize_search_text(signal) in normalized for signal in seller_default_signals):
                return user.crm_owner_name
        return None

    def _normalize_user_scoped_question_v2(self, question: str, owner_scope: str | None) -> str:
        if not owner_scope:
            return question
        normalized = self._normalize_search_text(question)
        if any(token in f" {normalized} " for token in [" mi ", " mis ", " yo ", "mias", "mios"]):
            return f"{question.strip()} del vendedor {owner_scope}"
        return question

    def _owner_alias_map(self) -> dict[str, str]:
        return {
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

    def _owner_scope_from_question(self, question: str) -> str | None:
        normalized = self._normalize_search_text(question)
        alias_map = {
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
        for alias, owner_name in alias_map.items():
            if alias in normalized:
                return owner_name
        return None

    def _mentioned_owner_names(self, question: str) -> list[str]:
        normalized = self._normalize_search_text(question)
        found: list[str] = []
        alias_map = {
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
        for alias, owner_name in alias_map.items():
            if alias in normalized and owner_name not in found:
                found.append(owner_name)
        return found

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
            if row.get("full_name") and normalized_term == self._normalize_search_text(row["full_name"]):
                score += 25

            if normalized_term and score >= 20:
                scored.append((score, row))

        scored.sort(key=lambda item: (-item[0], item[1].get("company_name") or item[1].get("full_name") or ""))
        return [row for _, row in scored[:10]]

    def _filtered_matches_for_entity(self, conn: sqlite3.Connection, term: str) -> list[dict[str, Any]]:
        normalized_term = self._normalize_search_text(self._clean_search_term(term))
        matches = self._entity_matches(conn, term)
        filtered: list[dict[str, Any]] = []
        for match in matches:
            fields = [
                match.get("company_name") or "",
                match.get("contact_name") or "",
                match.get("full_name") or "",
            ]
            joined = self._normalize_search_text(" ".join(fields))
            if normalized_term and normalized_term in joined:
                filtered.append(match)
        return filtered or matches[:3]

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

    def _contact_directory_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        matches = self._filtered_matches_for_entity(conn, term)
        notes = self._recent_notes_for_entity(conn, term)
        lines: list[str] = []
        seen: set[str] = set()

        for match in matches:
            label = match.get("contact_name") or match.get("full_name") or match.get("company_name") or "Sin nombre"
            email = match.get("email") or "Sin correo"
            phone = match.get("phone") or "Sin telefono"
            key = f"{label}|{email}|{phone}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- {label} | Correo: {email} | Telefono: {phone} | Responsable CRM: {match.get('owner_name') or 'Sin responsable'}"
            )

        for contact in self._emails_from_notes(conn, term, include_names=True):
            name, _, email = contact.partition(" | ")
            normalized = email.replace("Contacto en nota | ", "").strip()
            key = f"{name}|{normalized}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {name} | Correo: {normalized} | Telefono: Sin telefono | Fuente: nota CRM")

        if not lines:
            return f"No hay evidencia disponible sobre contactos de {term} en la información proporcionada."
        return f"Contactos de {term} disponibles:\n\n" + "\n".join(lines)

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
                    name = self._guess_contact_name_from_note_v2(content, cleaned_email) or "Contacto en nota"
                    extracted.append(f"{name} | {cleaned_email}")
                else:
                    extracted.append(cleaned_email)
        return extracted

    def _guess_contact_name_from_note_v2(self, content: str, email: str) -> str | None:
        patterns = [
            rf"contacto correcto de ([A-Za-zÁÉÍÓÚÑáéíóúñ ]+) \([^)]*\).*?{re.escape(email)}",
            rf"correo de ([A-Za-zÁÉÍÓÚÑáéíóúñ ]+).*?{re.escape(email)}",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return " ".join(match.group(1).split())
        local_name = email.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
        if local_name:
            return " ".join(part.capitalize() for part in local_name.split())
        return None

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

    def _owner_kpis(self, conn: sqlite3.Connection, owner_scope: str | None = None, window_days: int | None = None) -> dict[str, Any]:
        def where_clause(date_column: str) -> tuple[str, list[Any]]:
            filters: list[str] = []
            params_local: list[Any] = []
            if owner_scope:
                filters.append("owner_name = ?")
                params_local.append(owner_scope)
            if window_days:
                filters.append(f"date({date_column}) >= date('now', ?)")
                params_local.append(f"-{window_days} day")
            return (f"WHERE {' AND '.join(filters)}" if filters else "", params_local)

        lead_where, lead_params = where_clause("COALESCE(modified_time, created_time)")
        contact_where, contact_params = where_clause("COALESCE(modified_time, created_time)")
        note_where, note_params = where_clause("created_time")
        interaction_where, interaction_params = where_clause("interaction_at")
        task_where, task_params = where_clause("COALESCE(due_date, created_time)")

        leads = conn.execute(f"SELECT COUNT(*) FROM leads {lead_where}", lead_params).fetchone()[0]
        contacts = conn.execute(f"SELECT COUNT(*) FROM contacts {contact_where}", contact_params).fetchone()[0]
        notes = conn.execute(f"SELECT COUNT(*) FROM notes {note_where}", note_params).fetchone()[0]
        interactions = conn.execute(f"SELECT COUNT(*) FROM interactions {interaction_where}", interaction_params).fetchone()[0]
        open_tasks = conn.execute(
            f"SELECT COUNT(*) FROM tasks {task_where}{' AND ' if task_where else 'WHERE '}(status IS NULL OR lower(status) NOT IN ('completed', 'cancelled'))",
            task_params,
        ).fetchone()[0]
        last_activity_row = conn.execute(
            f"SELECT MAX(interaction_at) FROM interactions {interaction_where}",
            interaction_params,
        ).fetchone()
        return {
            "owner_scope": owner_scope,
            "leads": leads,
            "contacts": contacts,
            "notes": notes,
            "interactions": interactions,
            "open_tasks": open_tasks,
            "last_activity": last_activity_row[0] if last_activity_row else None,
        }

    def _global_kpis(self, conn: sqlite3.Connection, window_days: int | None = None) -> list[dict[str, Any]]:
        owners = [row["owner_name"] for row in self._owner_load(conn)]
        results: list[dict[str, Any]] = []
        for owner_name in owners:
            kpis = self._owner_kpis(conn, owner_name, window_days=window_days)
            if any(kpis[key] for key in ("leads", "contacts", "notes", "interactions", "open_tasks")):
                results.append({"owner_name": owner_name, **kpis})
        results.sort(key=lambda item: (-(item["interactions"] + item["notes"] + item["contacts"] + item["leads"]), item["owner_name"]))
        return results

    def _assigned_clients(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        if not owner_scope:
            return []
        query = """
        SELECT entity_type, company_name, contact_name, full_name, email, phone, owner_name, last_activity_time
        FROM (
            SELECT 'lead' AS entity_type, company_name, contact_name, full_name, email, phone, owner_name, last_activity_time
            FROM leads
            UNION ALL
            SELECT 'contact' AS entity_type, company_name, contact_name, full_name, email, phone, owner_name, last_activity_time
            FROM contacts
        )
        WHERE owner_name = ?
        ORDER BY COALESCE(last_activity_time, '') DESC, company_name ASC, full_name ASC
        LIMIT 50
        """
        return [dict(row) for row in conn.execute(query, (owner_scope,)).fetchall()]

    def _owner_comparison(self, conn: sqlite3.Connection, question: str) -> list[dict[str, Any]] | None:
        owners = self._mentioned_owner_names(question)
        if len(owners) < 2:
            return None
        results: list[dict[str, Any]] = []
        for owner_name in owners[:2]:
            kpis = self._owner_kpis(conn, owner_name)
            recent = self._recent_activity_by_owner(conn, owner_name)
            pending = self._pending_tasks(conn, owner_name)
            results.append(
                {
                    "owner_name": owner_name,
                    "kpis": kpis,
                    "recent_activity": recent[0] if recent else None,
                    "pending_count": len(pending),
                }
            )
        return results

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
          AND date(interaction_at, 'localtime') = date('now', 'localtime', ?)
        ORDER BY interaction_at DESC
        LIMIT 25
        """
        if owner_scope:
            query = """
            SELECT related_name, owner_name, source_type, interaction_at, status, summary
            FROM interactions
            WHERE interaction_at IS NOT NULL
              AND date(interaction_at, 'localtime') = date('now', 'localtime', ?)
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

    def _clean_search_term(self, value: str) -> str:
        cleaned = " ".join((value or "").strip(" ?.:,;").split())
        removable_prefixes = [
            "compara ",
            "comparar ",
            "comparativa ",
            "dame ",
            "solo ",
            "ok ",
            "cliente ",
            "prospecto ",
            "contacto ",
            "contactos ",
            "empresa ",
            "nombres ",
            "telefonos ",
            "telefonos y correos ",
            "telefonos y nombres ",
            "correos y nombres ",
            "de ",
            "del ",
        ]
        lowered = cleaned.lower()
        changed = True
        while changed and lowered:
            changed = False
            for prefix in removable_prefixes:
                if lowered.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    lowered = cleaned.lower()
                    changed = True
        return cleaned.strip()

    def _owner_scope_from_question(self, question: str) -> str | None:
        normalized = self._normalize_search_text(question)
        alias_map = {
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
        for alias, owner_name in alias_map.items():
            if alias in normalized:
                return owner_name
        return None

    def _mentioned_owner_names(self, question: str) -> list[str]:
        normalized = self._normalize_search_text(question)
        found: list[str] = []
        alias_map = {
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
        for alias, owner_name in alias_map.items():
            if alias in normalized and owner_name not in found:
                found.append(owner_name)
        return found

    def _field_values_for_entity(self, conn: sqlite3.Connection, term: str, field_name: str) -> list[str]:
        matches = self._filtered_matches_for_entity(conn, term)
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

    def _emails_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        rows = self._field_values_for_entity(conn, term, "email")
        note_emails = self._emails_from_notes(conn, term)
        values = sorted({*rows, *note_emails})
        return "\n".join(values) if values else "No encontre correos para ese cliente o prospecto."

    def _phones_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        matches = self._filtered_matches_for_entity(conn, term)
        lines: list[str] = []
        seen: set[str] = set()
        for match in matches:
            phone = (match.get("phone") or "").strip()
            if not phone:
                continue
            label = match.get("contact_name") or match.get("full_name") or match.get("company_name") or "Sin nombre"
            key = f"{label}|{phone}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{label} | {phone}")
        return "\n".join(lines) if lines else "No encontre telefonos para ese cliente o prospecto."

    def _contact_directory_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        matches = self._filtered_matches_for_entity(conn, term)
        lines: list[str] = []
        seen: set[str] = set()

        for match in matches:
            label = match.get("contact_name") or match.get("full_name") or match.get("company_name") or "Sin nombre"
            email = match.get("email") or "Sin correo"
            phone = match.get("phone") or "Sin telefono"
            key = f"{self._normalize_search_text(label)}|{email.lower()}|{phone}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- {label} | Correo: {email} | Telefono: {phone} | Responsable CRM: {match.get('owner_name') or 'Sin responsable'}"
            )

        for contact in self._emails_from_notes(conn, term, include_names=True):
            name, _, email = contact.partition(" | ")
            normalized_name = self._normalize_search_text(name)
            normalized_email = email.strip().lower()
            key = f"{normalized_name}|{normalized_email}|Sin telefono"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {name} | Correo: {normalized_email} | Telefono: Sin telefono | Fuente: nota CRM")

        if not lines:
            return f"No hay evidencia disponible sobre contactos de {term} en la informacion proporcionada."
        return f"Contactos de {term} disponibles:\n\n" + "\n".join(lines)

    def _interactions_on_relative_day(self, conn: sqlite3.Connection, days_ago: int, owner_scope: str | None = None) -> list[dict[str, Any]]:
        modifier = f"-{days_ago} day"
        query = """
        SELECT related_name, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE interaction_at IS NOT NULL
          AND substr(interaction_at, 1, 10) = date('now', 'localtime', ?)
        ORDER BY interaction_at DESC
        LIMIT 25
        """
        params: list[Any] = [modifier]
        if owner_scope:
            query = """
            SELECT related_name, owner_name, source_type, interaction_at, status, summary
            FROM interactions
            WHERE interaction_at IS NOT NULL
              AND substr(interaction_at, 1, 10) = date('now', 'localtime', ?)
              AND owner_name = ?
            ORDER BY interaction_at DESC
            LIMIT 25
            """
            params.append(owner_scope)
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _question_terms(self, question: str) -> list[str]:
        stopwords = {
            "de", "del", "la", "el", "los", "las", "para", "por", "que", "qué", "con", "sin",
            "como", "cómo", "una", "uno", "unas", "unos", "hoy", "ayer", "antier", "quien", "quién",
            "dame", "solo", "plan", "accion", "acción", "compara", "comparativa", "cliente", "prospecto",
            "contacto", "contactos", "vendedor", "vendedores", "kpi", "kpis", "global", "semana",
        }
        normalized = self._normalize_search_text(question)
        tokens = []
        for token in re.findall(r"[a-z0-9]{3,}", normalized):
            if token in stopwords:
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens[:8]

    def _document_search(self, conn: sqlite3.Connection, question: str) -> list[dict[str, Any]]:
        terms = self._question_terms(question)
        if not terms:
            return []

        clauses = " OR ".join(["lower(dc.content) LIKE lower(?)", "lower(d.file_name) LIKE lower(?)"] * len(terms))
        values: list[str] = []
        for term in terms:
            values.extend([f"%{term}%", f"%{term}%"])

        query = f"""
        SELECT d.file_name, dc.chunk_index, dc.content
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE {clauses}
        LIMIT 8
        """
        try:
            return [dict(row) for row in conn.execute(query, values).fetchall()]
        except sqlite3.OperationalError:
            return []

    def _raw_module_rows(self, module: str) -> list[dict[str, Any]]:
        cache_name = f"_cache_{module}"
        cached = getattr(self, cache_name, None)
        if cached is None:
            cached = load_json(RAW_MODULE_FILES[module])
            setattr(self, cache_name, cached)
        return cached

    def _owner_key(self, owner_name: str | None) -> str:
        return self._normalize_search_text(owner_name or "")

    def _owner_matches(self, candidate: str | None, owner_scope: str | None) -> bool:
        if not owner_scope:
            return True
        return self._owner_key(candidate) == self._owner_key(owner_scope)

    def _parse_datetime_safe(self, value: str | None) -> datetime | None:
        if not value:
            return None
        candidate = str(value).strip()
        if not candidate:
            return None
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _within_window(self, value: str | None, window_days: int | None) -> bool:
        if not window_days:
            return True
        parsed = self._parse_datetime_safe(value)
        if not parsed:
            return False
        return parsed.date() >= (datetime.now(parsed.tzinfo).date() - timedelta(days=window_days))

    def _matches_entity_snapshot(self, payload: dict[str, Any], term: str) -> bool:
        normalized_term = self._normalize_search_text(self._clean_search_term(term))
        fields = [
            payload.get("Empresa"),
            payload.get("Nombre_contacto"),
            payload.get("Full_Name"),
            payload.get("Last_Name"),
            ((payload.get("Account_Name") or {}) if isinstance(payload.get("Account_Name"), dict) else {}).get("name"),
            payload.get("Email"),
            payload.get("Phone"),
        ]
        haystack = self._normalize_search_text(" ".join(str(item or "") for item in fields))
        return bool(normalized_term and normalized_term in haystack)

    def _snapshot_contact_directory(self, term: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for module in ("leads", "contacts"):
            for payload in self._raw_module_rows(module):
                if not self._matches_entity_snapshot(payload, term):
                    continue
                owner = payload.get("Owner") or {}
                if not isinstance(owner, dict):
                    owner = {}
                label = payload.get("Nombre_contacto") or payload.get("Full_Name") or payload.get("Empresa") or payload.get("Last_Name") or "Sin nombre"
                company = payload.get("Empresa") or ((payload.get("Account_Name") or {}) if isinstance(payload.get("Account_Name"), dict) else {}).get("name")
                email = payload.get("Email") or "Sin correo"
                phone = payload.get("Phone") or payload.get("Mobile") or "Sin telefono"
                key = f"{self._normalize_search_text(label)}|{str(email).lower()}|{phone}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "label": label,
                        "company": company or "",
                        "email": email,
                        "phone": phone,
                        "owner_name": owner.get("name") or "Sin responsable",
                        "source": module,
                    }
                )

        for payload in self._raw_module_rows("notes"):
            parent = payload.get("Parent_Id") or {}
            parent_name = parent.get("name") if isinstance(parent, dict) else ""
            content = strip_html(payload.get("Note_Content"))
            if not self._matches_entity_snapshot({"Empresa": parent_name, "Full_Name": parent_name, "Email": content}, term):
                continue
            emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", content)
            for email in emails:
                clean_email = email.strip().strip(".,;:()[]<>").lower()
                name = self._guess_contact_name_from_note_v2(content, clean_email) or "Contacto en nota"
                key = f"{self._normalize_search_text(name)}|{clean_email}|Sin telefono"
                if key in seen:
                    continue
                seen.add(key)
                owner = payload.get("Owner") or {}
                if not isinstance(owner, dict):
                    owner = {}
                rows.append(
                    {
                        "label": name,
                        "company": parent_name or "",
                        "email": clean_email,
                        "phone": "Sin telefono",
                        "owner_name": owner.get("name") or "Sin responsable",
                        "source": "nota CRM",
                    }
                )
        return rows

    def _contact_directory_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        snapshot_rows = self._snapshot_contact_directory(term)
        if snapshot_rows:
            lines = []
            for row in snapshot_rows:
                suffix = f" | Empresa: {row['company']}" if row.get("company") else ""
                source = f" | Fuente: {row['source']}" if row.get("source") == "nota CRM" else ""
                lines.append(
                    f"- {row['label']} | Correo: {row['email']} | Telefono: {row['phone']} | Responsable CRM: {row['owner_name']}{suffix}{source}"
                )
            return f"Contactos de {term} disponibles:\n\n" + "\n".join(lines)
        return f"No hay evidencia disponible sobre contactos de {term} en la informacion proporcionada."

    def _emails_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        snapshot_rows = self._snapshot_contact_directory(term)
        emails = sorted({row["email"] for row in snapshot_rows if row.get("email") and row["email"] != "Sin correo"})
        return "\n".join(emails) if emails else "No encontre correos para ese cliente o prospecto."

    def _contact_points_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        snapshot_rows = self._snapshot_contact_directory(term)
        lines = []
        seen: set[str] = set()
        for row in snapshot_rows:
            email = row.get("email")
            if not email or email == "Sin correo":
                continue
            key = f"{self._normalize_search_text(row['label'])}|{email.lower()}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {row['label']} | {email}")
        return "\n".join(lines) if lines else "No encontre nombres y correos para ese cliente o prospecto."

    def _phones_for_entity(self, conn: sqlite3.Connection, term: str) -> str:
        snapshot_rows = self._snapshot_contact_directory(term)
        lines = []
        for row in snapshot_rows:
            if not row.get("phone") or row["phone"] == "Sin telefono":
                continue
            lines.append(f"{row['label']} | {row['phone']}")
        unique_lines = list(dict.fromkeys(lines))
        return "\n".join(unique_lines) if unique_lines else "No encontre telefonos para ese cliente o prospecto."

    def _assigned_clients(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        if not owner_scope:
            return []
        rows: list[dict[str, Any]] = []
        for module in ("leads", "contacts"):
            for payload in self._raw_module_rows(module):
                owner = payload.get("Owner") or {}
                if not isinstance(owner, dict) or not self._owner_matches(owner.get("name"), owner_scope):
                    continue
                rows.append(
                    {
                        "entity_type": "lead" if module == "leads" else "contact",
                        "company_name": payload.get("Empresa") or ((payload.get("Account_Name") or {}) if isinstance(payload.get("Account_Name"), dict) else {}).get("name"),
                        "contact_name": payload.get("Nombre_contacto"),
                        "full_name": payload.get("Full_Name") or payload.get("Last_Name"),
                        "email": payload.get("Email"),
                        "phone": payload.get("Phone") or payload.get("Mobile"),
                        "owner_name": owner.get("name"),
                        "last_activity_time": payload.get("Last_Activity_Time") or payload.get("Modified_Time") or payload.get("Created_Time"),
                    }
                )
        rows.sort(key=lambda item: item.get("last_activity_time") or "", reverse=True)
        return rows[:50]

    def _interactions_on_relative_day(self, conn: sqlite3.Connection, days_ago: int, owner_scope: str | None = None) -> list[dict[str, Any]]:
        target_date = datetime.now().date() - timedelta(days=days_ago)
        rows: list[dict[str, Any]] = []

        for payload in self._raw_module_rows("notes"):
            owner = payload.get("Owner") or {}
            created_time = payload.get("Created_Time")
            parsed = self._parse_datetime_safe(created_time)
            if not parsed or parsed.date() != target_date:
                continue
            if not isinstance(owner, dict) or not self._owner_matches(owner.get("name"), owner_scope):
                continue
            parent = payload.get("Parent_Id") or {}
            rows.append(
                {
                    "related_name": parent.get("name") if isinstance(parent, dict) else "Sin nombre",
                    "owner_name": owner.get("name") or "Sin responsable",
                    "source_type": "note",
                    "interaction_at": created_time,
                    "status": None,
                    "summary": strip_html(payload.get("Note_Content"))[:180],
                }
            )

        for payload in self._raw_module_rows("calls"):
            owner = payload.get("Owner") or {}
            start_time = payload.get("Call_Start_Time") or payload.get("Created_Time")
            parsed = self._parse_datetime_safe(start_time)
            if not parsed or parsed.date() != target_date:
                continue
            if not isinstance(owner, dict) or not self._owner_matches(owner.get("name"), owner_scope):
                continue
            who = payload.get("Who_Id") or {}
            rows.append(
                {
                    "related_name": who.get("name") if isinstance(who, dict) else "Sin nombre",
                    "owner_name": owner.get("name") or "Sin responsable",
                    "source_type": "call",
                    "interaction_at": start_time,
                    "status": payload.get("Call_Status"),
                    "summary": payload.get("Subject") or "",
                }
            )

        rows.sort(key=lambda item: item.get("interaction_at") or "", reverse=True)
        return rows[:25]

    def _interactions_by_owner(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        counters: dict[str, int] = {}
        for module, time_key in (("notes", "Created_Time"), ("calls", "Call_Start_Time"), ("events", "Created_Time")):
            for payload in self._raw_module_rows(module):
                owner = payload.get("Owner") or {}
                if not isinstance(owner, dict):
                    continue
                owner_name = owner.get("name")
                if not owner_name or not self._owner_matches(owner_name, owner_scope):
                    continue
                counters[owner_name] = counters.get(owner_name, 0) + 1
        rows = [{"owner_name": name, "total_interactions": total} for name, total in counters.items()]
        rows.sort(key=lambda item: (-item["total_interactions"], item["owner_name"]))
        return rows

    def _recent_activity_by_owner(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        since = datetime.now().date() - timedelta(days=30)
        stats: dict[str, dict[str, Any]] = {}
        for module, time_key in (("notes", "Created_Time"), ("calls", "Call_Start_Time"), ("events", "Created_Time")):
            for payload in self._raw_module_rows(module):
                owner = payload.get("Owner") or {}
                if not isinstance(owner, dict):
                    continue
                owner_name = owner.get("name")
                parsed = self._parse_datetime_safe(payload.get(time_key))
                if not owner_name or not parsed or parsed.date() < since:
                    continue
                if not self._owner_matches(owner_name, owner_scope):
                    continue
                entry = stats.setdefault(owner_name, {"owner_name": owner_name, "recent_interactions": 0, "latest_activity": None})
                entry["recent_interactions"] += 1
                latest = entry["latest_activity"]
                if latest is None or (payload.get(time_key) or "") > latest:
                    entry["latest_activity"] = payload.get(time_key)
        rows = list(stats.values())
        rows.sort(key=lambda item: (-item["recent_interactions"], item["latest_activity"] or ""))
        return rows

    def _owner_kpis(self, conn: sqlite3.Connection, owner_scope: str | None = None, window_days: int | None = None) -> dict[str, Any]:
        def count_people(module: str) -> int:
            total = 0
            for payload in self._raw_module_rows(module):
                owner = payload.get("Owner") or {}
                if not isinstance(owner, dict) or not self._owner_matches(owner.get("name"), owner_scope):
                    continue
                activity_time = payload.get("Modified_Time") or payload.get("Created_Time") or payload.get("Last_Activity_Time")
                if not self._within_window(activity_time, window_days):
                    continue
                total += 1
            return total

        def count_activity(module: str, time_key: str) -> tuple[int, str | None]:
            total = 0
            last_seen: str | None = None
            for payload in self._raw_module_rows(module):
                owner = payload.get("Owner") or {}
                if not isinstance(owner, dict) or not self._owner_matches(owner.get("name"), owner_scope):
                    continue
                activity_time = payload.get(time_key)
                if not self._within_window(activity_time, window_days):
                    continue
                total += 1
                if activity_time and (last_seen is None or activity_time > last_seen):
                    last_seen = activity_time
            return total, last_seen

        leads = count_people("leads")
        contacts = count_people("contacts")
        notes, notes_last = count_activity("notes", "Created_Time")
        calls, calls_last = count_activity("calls", "Call_Start_Time")
        events, events_last = count_activity("events", "Created_Time")

        open_tasks = 0
        task_last: str | None = None
        for payload in self._raw_module_rows("tasks"):
            owner = payload.get("Owner") or {}
            if not isinstance(owner, dict) or not self._owner_matches(owner.get("name"), owner_scope):
                continue
            status = (payload.get("Status") or "").lower()
            if status in {"completed", "cancelled"}:
                continue
            activity_time = payload.get("Due_Date") or payload.get("Created_Time") or payload.get("Modified_Time")
            if not self._within_window(activity_time, window_days):
                continue
            open_tasks += 1
            if activity_time and (task_last is None or activity_time > task_last):
                task_last = activity_time

        all_last = [value for value in (notes_last, calls_last, events_last, task_last) if value]
        return {
            "owner_scope": owner_scope,
            "leads": leads,
            "contacts": contacts,
            "notes": notes,
            "interactions": notes + calls + events,
            "open_tasks": open_tasks,
            "last_activity": max(all_last) if all_last else None,
        }

    def _global_kpis(self, conn: sqlite3.Connection, window_days: int | None = None) -> list[dict[str, Any]]:
        owners: set[str] = set()
        for module in ("leads", "contacts", "notes", "calls", "tasks", "events"):
            for payload in self._raw_module_rows(module):
                owner = payload.get("Owner") or {}
                if isinstance(owner, dict) and owner.get("name"):
                    owners.add(owner.get("name"))
        rows: list[dict[str, Any]] = []
        for owner_name in sorted(owners):
            kpis = self._owner_kpis(conn, owner_name, window_days=window_days)
            if any(kpis[key] for key in ("leads", "contacts", "notes", "interactions", "open_tasks")):
                rows.append({"owner_name": owner_name, **kpis})
        rows.sort(key=lambda item: (-(item["interactions"] + item["notes"] + item["contacts"] + item["leads"]), item["owner_name"]))
        return rows

    def _owner_scope_from_question(self, question: str) -> str | None:
        normalized = self._normalize_search_text(question)
        alias_map = {
            "eduardo": "Eduardo Valdez",
            "evaldez": "Eduardo Valdez",
            "ceo": "Eduardo Valdez",
            "pablo": "Pablo Melin Dorador",
            "pmelin": "Pablo Melin Dorador",
            "emmanuel": "Jesus Emmanuel Meza Guzm\u00e1n",
            "emeza": "Jesus Emmanuel Meza Guzm\u00e1n",
            "jesus emmanuel meza": "Jesus Emmanuel Meza Guzm\u00e1n",
            "jesus emmanuel meza guzman": "Jesus Emmanuel Meza Guzm\u00e1n",
        }
        for alias, owner_name in alias_map.items():
            if alias in normalized:
                return owner_name
        return None

    def _mentioned_owner_names(self, question: str) -> list[str]:
        normalized = self._normalize_search_text(question)
        found: list[str] = []
        alias_map = {
            "eduardo": "Eduardo Valdez",
            "evaldez": "Eduardo Valdez",
            "ceo": "Eduardo Valdez",
            "pablo": "Pablo Melin Dorador",
            "pmelin": "Pablo Melin Dorador",
            "emmanuel": "Jesus Emmanuel Meza Guzm\u00e1n",
            "emeza": "Jesus Emmanuel Meza Guzm\u00e1n",
            "jesus emmanuel meza": "Jesus Emmanuel Meza Guzm\u00e1n",
            "jesus emmanuel meza guzman": "Jesus Emmanuel Meza Guzm\u00e1n",
        }
        for alias, owner_name in alias_map.items():
            if alias in normalized and owner_name not in found:
                found.append(owner_name)
        return found

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

    def _format_owner_kpis(self, kpis: dict[str, Any], owner_scope: str | None = None, window_days: int | None = None) -> str:
        target = owner_scope or "todo el equipo"
        period = f" de los ultimos {window_days} dias" if window_days else ""
        return (
            f"KPIs de {target}{period}:\n"
            f"- Leads: {kpis['leads']}\n"
            f"- Contacts: {kpis['contacts']}\n"
            f"- Notas: {kpis['notes']}\n"
            f"- Interacciones: {kpis['interactions']}\n"
            f"- Pendientes abiertos: {kpis['open_tasks']}\n"
            f"- Ultima actividad registrada: {kpis['last_activity'] or 'sin dato'}"
        )

    def _format_global_kpis(self, rows: list[dict[str, Any]], window_days: int | None = None) -> str:
        if not rows:
            return "No encontre KPIs globales para el periodo solicitado."
        period = f" de la ultima semana" if window_days == 7 else ""
        lines = [f"KPI global{period} de todos los vendedores:", ""]
        for row in rows[:12]:
            lines.append(
                f"- {row['owner_name']} | Leads: {row['leads']} | Contacts: {row['contacts']} | "
                f"Notas: {row['notes']} | Interacciones: {row['interactions']} | Pendientes: {row['open_tasks']} | "
                f"Ultima actividad: {row['last_activity'] or 'sin dato'}"
            )
        return "\n".join(lines)

    def _format_assigned_clients(self, rows: list[dict[str, Any]], owner_scope: str | None = None) -> str:
        if not owner_scope:
            return "Necesito identificar al vendedor para mostrar su cartera."
        if not rows:
            return f"No encontre clientes o prospectos asignados a {owner_scope}."
        lines = [f"Clientes y prospectos asignados a {owner_scope}:", ""]
        lines.extend(
            f"- {row['company_name'] or row['full_name'] or 'Sin nombre'} | {row['entity_type']} | "
            f"Contacto: {row['contact_name'] or row['full_name'] or 'Sin contacto'} | "
            f"Correo: {row['email'] or 'Sin correo'} | Telefono: {row['phone'] or 'Sin telefono'} | "
            f"Ultima actividad: {row['last_activity_time'] or 'sin dato'}"
            for row in rows
        )
        return "\n".join(lines)

    def _format_owner_comparison(self, rows: list[dict[str, Any]]) -> str:
        if len(rows) < 2:
            return "No hay evidencia suficiente para comparar a esos vendedores."
        first, second = rows[0], rows[1]

        def score(item: dict[str, Any]) -> int:
            recent = item.get("recent_activity") or {}
            return (
                item["kpis"]["leads"]
                + item["kpis"]["contacts"]
                + item["kpis"]["notes"]
                + item["kpis"]["interactions"]
                + (recent.get("recent_interactions") or 0)
            )

        first_score = score(first)
        second_score = score(second)
        leader = first if first_score >= second_score else second
        trailer = second if leader is first else first

        def block(item: dict[str, Any]) -> str:
            recent = item.get("recent_activity") or {}
            return (
                f"{item['owner_name']}:\n"
                f"- Leads: {item['kpis']['leads']}\n"
                f"- Contacts: {item['kpis']['contacts']}\n"
                f"- Notas: {item['kpis']['notes']}\n"
                f"- Interacciones: {item['kpis']['interactions']}\n"
                f"- Actividad reciente (30 dias): {recent.get('recent_interactions', 0)}\n"
                f"- Pendientes abiertos: {item['pending_count']}\n"
                f"- Ultima actividad: {item['kpis']['last_activity'] or 'sin dato'}"
            )

        return (
            "Comparativa de vendedores:\n\n"
            f"{block(first)}\n\n"
            f"{block(second)}\n\n"
            f"Conclusion:\n{leader['owner_name']} muestra mayor traccion comercial que {trailer['owner_name']} "
            f"porque acumula mas cartera, notas e interacciones registradas."
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
- Si hay fragmentos de PDFs internos relevantes, usalos y menciona el nombre del archivo en la respuesta.

Modo detectado: {intent.mode}
Pregunta: {question}

Evidencia:
{evidence}
        """.strip()

    def _build_prompt(self, question: str, intent: QuestionIntent, evidence: dict[str, Any]) -> str:
        document_chunks = evidence.get("document_chunks") or []
        document_hint = ""
        if document_chunks:
            doc_lines = []
            for chunk in document_chunks[:4]:
                preview = " ".join((chunk.get("content") or "").split())[:220]
                doc_lines.append(f"- {chunk.get('file_name')} :: {preview}")
            document_hint = "\nDocumentos internos relevantes:\n" + "\n".join(doc_lines)

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
- Si hay usuario activo, interpreta las preguntas en primera persona según su perfil salvo que se mencione otro vendedor explícitamente.
- Si hay fragmentos de PDFs internos relevantes, úsalos y menciona el nombre del archivo en la respuesta.
- Si existe una respuesta directa previa en la evidencia, úsala como base, pero enriquécela con notas y PDFs cuando el modo sea analysis o hybrid.

Modo detectado: {intent.mode}
Pregunta: {question}
{document_hint}

Evidencia:
{evidence}
        """.strip()

    def _clean_search_term(self, value: str) -> str:
        cleaned = " ".join((value or "").strip(" ?.:,;").split())
        removable_prefixes = [
            "compara ",
            "comparar ",
            "comparativa ",
            "dame ",
            "solo ",
            "ok ",
            "cliente ",
            "clientes ",
            "prospecto ",
            "prospectos ",
            "contacto ",
            "contactos ",
            "empresa ",
            "nombres ",
            "telefonos ",
            "telefonos y correos ",
            "telefonos y nombres ",
            "correos y nombres ",
            "de ",
            "del ",
        ]
        lowered = cleaned.lower()
        changed = True
        while changed and lowered:
            changed = False
            for prefix in removable_prefixes:
                if lowered.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    lowered = cleaned.lower()
                    changed = True
        return cleaned.strip()

    def _document_search(self, conn: sqlite3.Connection, question: str) -> list[dict[str, Any]]:
        terms = self._question_terms(question)
        if not terms:
            return []

        clauses = " OR ".join(["lower(dc.content) LIKE lower(?)", "lower(d.file_name) LIKE lower(?)"] * len(terms))
        values: list[str] = []
        for term in terms:
            values.extend([f"%{term}%", f"%{term}%"])

        query = f"""
        SELECT d.file_name, dc.chunk_index, dc.content
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE {clauses}
        LIMIT 8
        """
        try:
            return [dict(row) for row in conn.execute(query, values).fetchall()]
        except sqlite3.OperationalError:
            return []
