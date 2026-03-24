import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
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


@dataclass
class EntityResolution:
    canonical_name: str | None
    aliases: list[str]
    confidence: float = 0.0


class SalesAssistantService:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or str(WAREHOUSE_DB)
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def answer_question(self, question: str, user: AppUser | None = None) -> AssistantResponse:
        subquestions = self._split_compound_question(question)
        if len(subquestions) > 1:
            subquestions = self._contextualize_subquestions(question, subquestions, user=user)
            subresponses = [self._answer_single_question(subquestion, user=user) for subquestion in subquestions]
            combined_answer = self._combine_subresponses(question, subquestions, subresponses)
            combined_sources = sorted({source for response in subresponses for source in response.sources})
            return AssistantResponse(
                mode="hybrid",
                sources=combined_sources,
                used_web=False,
                answer=combined_answer,
                evidence={"subquestions": subquestions, "subresponses": [response.evidence for response in subresponses]},
            )
        return self._answer_single_question(question, user=user)

    def _answer_single_question(self, question: str, user: AppUser | None = None) -> AssistantResponse:
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

    def _split_compound_question(self, question: str) -> list[str]:
        normalized = self._normalize_search_text(question)
        if "analiza mis notas y arma un plan para hoy" in normalized:
            return [question]
        if "conclusion y luego evidencia" in normalized or "conclusion primero y luego evidencia" in normalized:
            return [question]
        question_parts = [
            part.strip(" ,.;:-")
            for part in re.split(r"\?\s+(?=(?:que|cuantos|cuantas|como|dame|hazme|analiza|resume|redacta|escribe)\b)", question.strip(), flags=re.IGNORECASE)
            if part and part.strip(" ,.;:-")
        ]
        if len(question_parts) > 1:
            return question_parts
        if not any(token in normalized for token in [" y luego ", " despues ", " después ", " y redacta ", " y proponme ", " detecta riesgo y redacta ", " y escribe ", " y arma ", " y hazme ", " y fabrica "]):
            return [question]
        pattern = re.compile(
            r"\s*,?\s*(?:y luego|despues|después|y redacta|y escribe|y arma|y hazme|y fabrica|y proponme)\s+",
            flags=re.IGNORECASE,
        )
        parts = pattern.split(question.strip())
        cleaned = [part.strip(" ,.;:-") for part in parts if part and part.strip(" ,.;:-")]
        return cleaned or [question]

    def _combine_subresponses(
        self,
        question: str,
        subquestions: list[str],
        subresponses: list[AssistantResponse],
    ) -> str:
        if not subresponses:
            return "No encontre evidencia suficiente para responder."
        if len(subresponses) <= 2:
            merged: list[str] = []
            seen: set[str] = set()
            for response in subresponses:
                answer = response.answer.strip()
                if not answer:
                    continue
                key = self._normalize_search_text(answer)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(answer)
            return "\n\n".join(merged).strip() or "No encontre evidencia suficiente para responder."
        lines = [f"Respuesta integrada para: {question}", ""]
        for index, (subquestion, subresponse) in enumerate(zip(subquestions, subresponses), start=1):
            lines.append(f"{index}. {subquestion}")
            lines.append(subresponse.answer.strip())
            lines.append("")
        return "\n".join(lines).strip()

    def _contextualize_subquestions(self, original_question: str, subquestions: list[str], user: AppUser | None = None) -> list[str]:
        carried_entity = self._extract_entity_hint(original_question)
        owner_scope = self._resolve_owner_scope(original_question, user)
        relative_entity = self._top_priority_entity_for_owner(owner_scope) if owner_scope else None
        contextualized: list[str] = []
        for subquestion in subquestions:
            extracted = self._extract_entity_hint(subquestion)
            normalized = self._normalize_search_text(subquestion)
            if extracted and not self._looks_like_instruction_phrase(normalized, extracted) and not self._is_relative_reference(extracted):
                contextualized.append(subquestion)
                continue
            if any(token in normalized for token in ["la principal", "el principal", "el mejor", "la mejor"]) and relative_entity:
                carried = relative_entity
            else:
                carried = carried_entity
            if not carried:
                contextualized.append(subquestion)
                continue
            if any(token in normalized for token in ["correo", "mail", "email", "mensaje", "whatsapp", "seguimiento", "redacta", "redactame", "escribe"]):
                rewritten = subquestion.strip()
                original_normalized = self._normalize_search_text(original_question)
                if rewritten.lower().startswith("seguimiento para") and "redacta seguimiento" in original_normalized:
                    rewritten = f"redacta {rewritten}"
                if rewritten.lower().startswith("un correo para") or rewritten.lower().startswith("correo para"):
                    rewritten = f"redactame {rewritten}"
                if rewritten.lower().startswith("mensaje para") or rewritten.lower().startswith("un mensaje para"):
                    rewritten = f"armame {rewritten}"
                contextualized.append(f"{rewritten} para {carried}" if f"para {carried}".lower() not in rewritten.lower() else rewritten)
            elif any(token in normalized for token in ["ese nombre", "esa cuenta", "ese cliente", "ese prospecto", "para el", "para ella"]):
                rewritten = subquestion.strip()
                for marker in ["ese nombre", "esa cuenta", "ese cliente", "ese prospecto", "para el", "para ella"]:
                    rewritten = re.sub(marker, carried, rewritten, flags=re.IGNORECASE)
                if carried.lower() not in rewritten.lower():
                    rewritten = f"{rewritten} con {carried}"
                contextualized.append(rewritten)
            elif "responderlas" in normalized:
                contextualized.append(f"objeciones probables y como responderlas de {carried}")
            elif any(token in normalized for token in ["riesgo", "riesgos", "objecion", "objeciones", "resume", "resumen", "cuenta"]):
                contextualized.append(f"{subquestion.strip()} de {carried}")
            else:
                contextualized.append(subquestion)
        return contextualized

    def _looks_like_instruction_phrase(self, normalized_question: str, extracted_hint: str) -> bool:
        if extracted_hint != normalized_question:
            return False
        instruction_markers = [
            "redacta",
            "redactame",
            "escribe",
            "correo",
            "mensaje",
            "whatsapp",
            "seguimiento",
            "resume",
            "resumen",
            "cuenta",
            "riesgo",
            "objeciones",
            "propuesta",
            "argumentos",
            "la principal",
            "el principal",
            "el mejor",
            "la mejor",
        ]
        return any(marker in normalized_question for marker in instruction_markers)

    def _is_relative_reference(self, extracted_hint: str) -> bool:
        return extracted_hint in {"la principal", "el principal", "el mejor", "la mejor"}

    def _top_priority_entity_for_owner(self, owner_scope: str | None) -> str | None:
        if not owner_scope:
            return None
        rows = self.get_priority_followups(owner=owner_scope, limit=1)
        if not rows:
            return None
        return rows[0].get("related_name")

    def _should_prefer_direct_answer(self, intent: QuestionIntent, evidence: dict[str, Any]) -> bool:
        if intent.mode == "data" and not any(
            [
                intent.asks_for_sales_draft,
                intent.asks_for_action_plan,
                intent.asks_for_sales_material,
                intent.asks_for_formatted_output,
                intent.asks_for_multi_step,
            ]
        ):
            return True
        structured_keys = [
            "entity_brief",
            "owner_brief",
            "team_brief",
            "sales_draft",
            "action_plan",
            "sales_material",
            "today_call_list",
            "comparison_candidates",
            "owner_comparison",
            "owner_load",
            "ranked_accounts",
            "owner_kpis",
            "global_kpis",
            "assigned_clients",
            "entity_count_summary",
            "owner_client_count",
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
        normalized_question = self._normalize_search_text(question)
        evidence: dict[str, Any] = {
            "sources": ["warehouse.db"],
            "entity_hint": entity_term,
            "direct_answer": None,
        }

        with self._connect() as conn:
            if owner_scope:
                evidence["owner_scope"] = owner_scope
            if entity_term:
                resolution = self._resolve_entity(conn, entity_term, owner_scope=owner_scope)
                evidence["entity_resolution"] = {
                    "canonical_name": resolution.canonical_name,
                    "aliases": resolution.aliases,
                    "confidence": resolution.confidence,
                }
                evidence["matches"] = self._entity_matches(conn, entity_term, owner_scope=owner_scope, resolution=resolution)
                evidence["recent_interactions"] = self._recent_interactions_for_entity(conn, entity_term, owner_scope=owner_scope, resolution=resolution)
                evidence["recent_notes"] = self._recent_notes_for_entity(conn, entity_term, owner_scope=owner_scope, resolution=resolution)
                evidence["contact_rows"] = [row.__dict__ for row in self._contact_rows_for_entity(conn, entity_term, owner_scope=owner_scope, resolution=resolution)]
                if not (
                    evidence.get("matches")
                    or evidence.get("recent_interactions")
                    or evidence.get("recent_notes")
                    or evidence.get("contact_rows")
                ):
                    evidence["entity_suggestions"] = self._entity_suggestions(conn, entity_term, owner_scope=owner_scope)
            entity_has_evidence = bool(
                evidence.get("matches")
                or evidence.get("recent_interactions")
                or evidence.get("recent_notes")
                or evidence.get("contact_rows")
            )
            asks_for_count = any(token in normalized_question for token in ["cuantos", "cuantas", "cuanto", "cantidad", "numero de", "número de"])
            asks_for_rank = any(token in normalized_question for token in ["mejor", "mejores", "top", "principal", "principales"]) and any(
                token in normalized_question for token in ["cliente", "clientes", "prospecto", "prospectos", "cuenta", "cuentas", "lead", "leads"]
            )

            if asks_for_rank and owner_scope:
                limit = self._extract_rank_limit(normalized_question)
                evidence["ranked_accounts"] = self._owner_ranked_accounts(conn, owner_scope, limit=limit)
                evidence["direct_answer"] = self._format_owner_ranked_accounts(evidence["ranked_accounts"], owner_scope, limit)
            elif asks_for_count and entity_term and entity_has_evidence:
                evidence["entity_count_summary"] = self._entity_count_summary(conn, entity_term, owner_scope=owner_scope)
                evidence["direct_answer"] = self._format_entity_count_summary(evidence["entity_count_summary"])
            elif asks_for_count and owner_scope and any(token in normalized_question for token in ["cliente", "clientes", "prospecto", "prospectos", "lead", "leads", "cuenta", "cuentas"]):
                evidence["owner_client_count"] = self._owner_client_count_summary(conn, owner_scope)
                evidence["direct_answer"] = self._format_owner_client_count_summary(evidence["owner_client_count"], owner_scope)
            elif intent.asks_for_action_plan:
                if entity_term and entity_has_evidence:
                    evidence["entity_brief"] = self._entity_brief(conn, entity_term, owner_scope=owner_scope, question=question)
                    evidence["action_plan"] = self._entity_action_plan(evidence["entity_brief"])
                    evidence["direct_answer"] = self._format_action_plan(evidence["action_plan"])
                else:
                    effective_owner = owner_scope or self._owner_scope_from_question(conn, question)
                    evidence["owner_brief"] = self._owner_brief(conn, effective_owner)
                    evidence["action_plan"] = self._owner_action_plan(evidence["owner_brief"], effective_owner)
                    evidence["direct_answer"] = self._format_action_plan(evidence["action_plan"])
            elif intent.asks_for_team_brief:
                evidence["team_brief"] = self._team_brief(conn)
                evidence["direct_answer"] = self._format_team_brief(evidence["team_brief"])
            elif intent.asks_for_owner_brief:
                effective_owner = owner_scope or self._owner_scope_from_question(conn, question)
                evidence["owner_brief"] = self._owner_brief(conn, effective_owner)
                evidence["direct_answer"] = self._format_owner_brief(evidence["owner_brief"], effective_owner)
            elif entity_term and intent.asks_for_sales_draft:
                evidence["entity_brief"] = self._entity_brief(conn, entity_term, owner_scope=owner_scope, question=question)
                evidence["sales_draft"] = self._sales_draft(question, evidence["entity_brief"])
                evidence["direct_answer"] = self._format_sales_draft(evidence["sales_draft"])
            elif entity_term and intent.asks_for_sales_material:
                evidence["entity_brief"] = self._entity_brief(conn, entity_term, owner_scope=owner_scope, question=question)
                evidence["sales_material"] = self._sales_material(question, evidence["entity_brief"])
                evidence["direct_answer"] = self._format_sales_material(evidence["sales_material"])
            elif entity_term and intent.asks_for_client_brief:
                evidence["entity_brief"] = self._entity_brief(conn, entity_term, owner_scope=owner_scope, question=question)
                evidence["direct_answer"] = self._format_entity_brief(evidence["entity_brief"])
            elif entity_term and (intent.asks_for_contact_directory or (intent.asks_for_names and intent.asks_for_phones)):
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
                evidence["direct_answer"] = self._format_yesterday_contacts(conn, evidence["yesterday_contacts"], owner_scope)
            elif intent.asks_for_day_before_yesterday_contacts:
                evidence["day_before_yesterday_contacts"] = self._interactions_on_relative_day(conn, 2, owner_scope)
                evidence["direct_answer"] = self._format_interaction_list(evidence["day_before_yesterday_contacts"], "No encontre contactos registrados antier.")
            elif intent.asks_for_latest_contacted:
                evidence["latest_contacted"] = self._latest_contacted(conn, owner_scope)
                evidence["direct_answer"] = self._format_latest_contacted(evidence["latest_contacted"])
            elif intent.asks_for_latest_note:
                evidence["latest_note"] = self._latest_note(conn, owner_scope)
                evidence["direct_answer"] = self._format_latest_note(evidence["latest_note"], owner_scope)
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
                owner_names = self._mentioned_owner_names(question)
                normalized_question = self._normalize_search_text(question)
                repeated_owner = next(
                    (
                        owner_name
                        for owner_name in owner_names
                        if normalized_question.count(self._normalize_search_text(owner_name)) >= 2
                    ),
                    None,
                )
                if repeated_owner:
                    evidence["direct_answer"] = (
                        f"La comparacion apunta al mismo vendedor ({repeated_owner}). "
                        "Pide dos vendedores distintos para contrastar actividad, cartera y pendientes."
                    )
                elif len(owner_names) >= 2:
                    if owner_names[0] == owner_names[1]:
                        evidence["direct_answer"] = (
                            f"La comparacion apunta al mismo vendedor ({owner_names[0]}). "
                            "Pide dos vendedores distintos para contrastar actividad, cartera y pendientes."
                        )
                    else:
                        evidence["owner_comparison"] = self._owner_comparison(conn, owner_names[:2])
                        evidence["direct_answer"] = (
                            self._format_owner_comparison(evidence["owner_comparison"])
                            if intent.mode == "data"
                            else self._format_owner_comparison_analysis(evidence["owner_comparison"])
                        )
                else:
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
                elif entity_term and entity_has_evidence and not owner_scope:
                    evidence["entity_kpis"] = self._entity_kpis(conn, entity_term)
                    evidence["direct_answer"] = self._format_entity_kpis(evidence["entity_kpis"], window_days)
                else:
                    evidence["owner_kpis"] = self._owner_kpis(conn, owner_scope, window_days)
                    evidence["direct_answer"] = self._format_owner_kpis(evidence["owner_kpis"], owner_scope, window_days)
            elif intent.asks_for_risks and entity_term:
                evidence["risk_profile"] = self._risk_profile(conn, entity_term, owner_scope)
                evidence["direct_answer"] = self._format_risk_profile(evidence["risk_profile"], entity_term)
            elif entity_term and entity_has_evidence:
                evidence["entity_brief"] = self._entity_brief(conn, entity_term, owner_scope=owner_scope, question=question)
                evidence["direct_answer"] = self._format_entity_brief(evidence["entity_brief"])
            elif entity_term and evidence.get("entity_suggestions"):
                evidence["direct_answer"] = self._format_entity_suggestions(entity_term, evidence["entity_suggestions"])

            evidence["document_chunks"] = self._document_search(conn, question, entity_term)
            if evidence["document_chunks"]:
                evidence["sources"].extend(sorted({chunk["file_name"] for chunk in evidence["document_chunks"]}))

        return evidence

    def _extract_entity_hint(self, question: str) -> str | None:
        normalized = self._normalize_search_text(question)
        sales_patterns = [
            r"(?:mandarle|enviarle|escribirle)\s+a\s+(.+?)(?:,|$)",
            r"(?:correo|mail|email|mensaje)\s+(?:para|a)\s+(.+?)(?:,|$)",
            r"(?:seguimiento|correo|mensaje|propuesta)\s+para\s+(.+?)(?:,|$)",
            r"(?:llamada|reunion|reunion)\s+con\s+(.+?)(?:,|$)",
            r"(?:esta semana|hoy|manana|mañana)\s+con\s+(.+?)(?:,|$)",
        ]
        for pattern in sales_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                candidate = self._clean_search_term(match.group(1).strip(" ?.:"))
                if candidate and not self._is_noise_entity_hint(candidate):
                    return candidate
        if " de " in normalized and any(token in normalized for token in ["correo", "telefono", "numero", "nombre", "contacto"]):
            trailing = self._clean_search_term(normalized.rsplit(" de ", 1)[-1].strip(" ?.:"))
            if trailing and trailing not in {"la semana", "semana"} and not self._is_noise_entity_hint(trailing):
                return trailing
        specific_patterns = [
            r"kpi(?:s)?\s+(.+?)\s+de\s+la\s+semana$",
            r"que sigue con\s+(.+)$",
            r"que objeciones?(?:\s+hay)?\s+(?:en|de)\s+(.+)$",
            r"que objeciones?\s+tiene\s+(.+)$",
            r"hazme un resumen ejecutivo de\s+(.+)$",
            r"resumen(?: comercial)? de\s+(.+)$",
            r"resumeme\s+(.+?)\s+en\s+contexto,\s*riesgo\s+y\s+siguiente\s+paso$",
            r"resume la cuenta(?: de)?\s+(.+)$",
            r"si entro a una llamada(?: en 5 minutos)? con\s+(.+)$",
            r"antes de una reunion con\s+(.+)$",
            r"(?:argumentos de venta|propuesta comercial breve|bullets de valor|beneficios(?: de nuestros servicios| de nuestros productos)?|speech de 30 segundos|mini agenda de reunion|preguntas de descubrimiento)\s+(?:para|de|con)\s+(.+)$",
            r"(?:que haria|que harías|que harias)\s+hoy,\s*manana?\s+y\s+esta\s+semana\s+con\s+(.+)$",
            r"(?:dame|quiero|necesito)\s+(?:los\s+)?nombres(?:\s*,\s*|\s+y\s+)?(?:correos?|emails?)(?:\s*,\s*|\s+y\s+)?(?:numeros?|telefonos?)\s+de\s+(.+)$",
            r"(?:dame|quiero|necesito)\s+(?:los\s+)?(?:nombres?|correos?|emails?|numeros?|telefonos?)\s+de\s+(.+)$",
        ]
        specific_patterns.append(r"(?:registrados?|registradas?|dados de alta)\s+(?:con|a nombre de)\s+(.+)$")
        for pattern in specific_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                candidate = self._clean_search_term(match.group(1).strip(" ?.:"))
                if candidate and not self._is_noise_entity_hint(candidate):
                    return candidate
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
                if candidate and not self._is_noise_entity_hint(candidate):
                    return candidate
        if ":" in normalized:
            trailing = self._clean_search_term(normalized.rsplit(":", 1)[-1].strip(" ?.:"))
            if trailing and not self._is_noise_entity_hint(trailing):
                return trailing
        if " de " in normalized and any(token in normalized for token in ["correo", "telefono", "numero", "nombre", "contacto", "kpi"]):
            trailing = self._clean_search_term(normalized.rsplit(" de ", 1)[-1].strip(" ?.:"))
            if trailing and trailing not in {"la semana", "semana"} and not self._is_noise_entity_hint(trailing):
                return trailing
        generic_questions = {
            "kpi global",
            "kpis globales",
            "kpi mio de la semana",
            "mis kpis",
            "kpi semanal",
            "ultima nota agregada",
            "dame clientes calientes y clientes frios",
            "dime todo lo que debo saber de mis contactos o leads",
            "que oportunidades tengo mas calientes",
            "donde estoy dejando dinero en la mesa",
            "analiza mis notas y arma un plan para hoy",
            "en base a mis notas, que me recomiendas hacer hoy",
            "dame primero conclusion y luego evidencia sobre mi cartera",
        }
        if normalized and normalized not in generic_questions and len(normalized.split()) <= 5:
            candidate = self._clean_search_term(normalized)
            if not self._is_noise_entity_hint(candidate):
                return candidate
        return None

    def _is_noise_entity_hint(self, candidate: str) -> bool:
        return candidate in {
            "alta",
            "de alta",
            "baja",
            "primero conclusion",
            "hoy",
            "ayer",
            "antier",
            "anteayer",
            "la semana",
            "semana",
            "el mes",
            "mes",
            "ese nombre",
        }

    def _resolve_entity(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> EntityResolution:
        cleaned = self._clean_search_term(term)
        if not cleaned:
            return EntityResolution(canonical_name=None, aliases=[])

        candidate_rows: list[str] = []
        queries = [
            ("SELECT DISTINCT company_name AS name FROM leads WHERE company_name IS NOT NULL", []),
            ("SELECT DISTINCT company_name AS name FROM contacts WHERE company_name IS NOT NULL", []),
            ("SELECT DISTINCT related_name AS name FROM interactions WHERE related_name IS NOT NULL", []),
            ("SELECT DISTINCT parent_name AS name FROM notes WHERE parent_name IS NOT NULL", []),
        ]
        for query, params in queries:
            try:
                candidate_rows.extend(
                    row["name"]
                    for row in conn.execute(query, params).fetchall()
                    if row["name"] and str(row["name"]).strip()
                )
            except sqlite3.OperationalError:
                continue

        term_tokens = self._entity_tokens(cleaned)
        scored: list[tuple[int, str]] = []
        for raw_name in dict.fromkeys(candidate_rows):
            normalized_name = self._clean_search_term(str(raw_name))
            if not normalized_name:
                continue
            score = 0
            ratio = SequenceMatcher(None, cleaned, normalized_name).ratio()
            if normalized_name == cleaned:
                score += 120
            if cleaned in normalized_name or normalized_name in cleaned:
                score += 70
            name_tokens = self._entity_tokens(normalized_name)
            overlap = len(term_tokens & name_tokens)
            if overlap:
                score += overlap * 25
            if term_tokens and term_tokens.issubset(name_tokens):
                score += 15
            if normalized_name.startswith(cleaned) or cleaned.startswith(normalized_name):
                score += 10
            score += int(ratio * 45)
            if score > 0:
                scored.append((score, str(raw_name)))

        if not scored:
            return EntityResolution(canonical_name=cleaned, aliases=[cleaned], confidence=0.0)

        scored.sort(key=lambda item: (-item[0], item[1]))
        top_score = scored[0][0]
        top_name = scored[0][1]
        top_ratio = SequenceMatcher(None, cleaned, self._clean_search_term(top_name)).ratio()
        if top_score < 55 and top_ratio < 0.72:
            return EntityResolution(canonical_name=cleaned, aliases=[cleaned], confidence=top_ratio)
        aliases = [name for score, name in scored if score >= max(45, top_score - 20)]
        canonical_name = aliases[0] if aliases else cleaned
        cleaned_aliases = list(dict.fromkeys([canonical_name, *aliases, cleaned]))
        return EntityResolution(canonical_name=canonical_name, aliases=cleaned_aliases[:6], confidence=top_ratio)

    def _entity_patterns(self, resolution: EntityResolution) -> list[str]:
        aliases = resolution.aliases or ([resolution.canonical_name] if resolution.canonical_name else [])
        patterns = [f"%{self._clean_search_term(alias)}%" for alias in aliases if alias]
        return patterns or ["%"]

    def _entity_suggestions(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None, limit: int = 3) -> list[str]:
        cleaned = self._clean_search_term(term)
        if not cleaned:
            return []
        candidate_rows: list[str] = []
        queries = [
            "SELECT DISTINCT company_name AS name FROM leads WHERE company_name IS NOT NULL",
            "SELECT DISTINCT company_name AS name FROM contacts WHERE company_name IS NOT NULL",
            "SELECT DISTINCT related_name AS name FROM interactions WHERE related_name IS NOT NULL",
            "SELECT DISTINCT parent_name AS name FROM notes WHERE parent_name IS NOT NULL",
        ]
        for query in queries:
            try:
                candidate_rows.extend(
                    row["name"]
                    for row in conn.execute(query).fetchall()
                    if row["name"] and str(row["name"]).strip()
                )
            except sqlite3.OperationalError:
                continue
        term_tokens = self._entity_tokens(cleaned)
        scored: list[tuple[float, str]] = []
        for raw_name in dict.fromkeys(candidate_rows):
            normalized_name = self._clean_search_term(str(raw_name))
            if not normalized_name:
                continue
            ratio = SequenceMatcher(None, cleaned, normalized_name).ratio()
            overlap = len(term_tokens & self._entity_tokens(normalized_name))
            if ratio >= 0.42 or overlap >= 1:
                scored.append((ratio + overlap * 0.15, str(raw_name)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored[:limit]]

    def _entity_tokens(self, value: str) -> set[str]:
        stopwords = {
            "grupo",
            "empresa",
            "empresarial",
            "sa",
            "cv",
            "de",
            "del",
            "la",
            "el",
            "los",
            "las",
            "y",
            "servicios",
            "transportes",
            "transporte",
            "logistica",
        }
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", self._normalize_search_text(value))
            if len(token) >= 3 and token not in stopwords
        }
        return tokens

    def _entity_matches(
        self,
        conn: sqlite3.Connection,
        term: str,
        owner_scope: str | None = None,
        resolution: EntityResolution | None = None,
    ) -> list[dict[str, Any]]:
        resolution = resolution or self._resolve_entity(conn, term, owner_scope=owner_scope)
        patterns = self._entity_patterns(resolution)
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
            {clauses}
        )
        """
        field_clauses = []
        params: list[Any] = []
        for pattern in patterns:
            for field in ["company_name", "contact_name", "full_name", "email", "phone"]:
                field_clauses.append(f"lower(coalesce({field}, '')) LIKE lower(?)")
                params.append(pattern)
        query = query.format(clauses=" OR ".join(field_clauses))
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY last_activity_time DESC, company_name, contact_name LIMIT 25"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _recent_interactions_for_entity(
        self,
        conn: sqlite3.Connection,
        term: str,
        owner_scope: str | None = None,
        resolution: EntityResolution | None = None,
    ) -> list[dict[str, Any]]:
        resolution = resolution or self._resolve_entity(conn, term, owner_scope=owner_scope)
        patterns = self._entity_patterns(resolution)
        query = """
        SELECT related_name, related_module, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE ({clauses})
        """
        params: list[Any] = []
        for pattern in patterns:
            params.append(pattern)
        query = query.format(clauses=" OR ".join("lower(coalesce(related_name, '')) LIKE lower(?)" for _ in patterns))
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY interaction_at DESC LIMIT 12"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _recent_notes_for_entity(
        self,
        conn: sqlite3.Connection,
        term: str,
        owner_scope: str | None = None,
        resolution: EntityResolution | None = None,
    ) -> list[dict[str, Any]]:
        resolution = resolution or self._resolve_entity(conn, term, owner_scope=owner_scope)
        patterns = self._entity_patterns(resolution)
        query = """
        SELECT parent_name, parent_module, owner_name, created_time, title, content_text
        FROM notes
        WHERE (
            {clauses}
        )
        """
        params: list[Any] = []
        note_clauses = []
        for pattern in patterns:
            note_clauses.append("lower(coalesce(parent_name, '')) LIKE lower(?)")
            params.append(pattern)
            note_clauses.append("lower(coalesce(content_text, '')) LIKE lower(?)")
            params.append(pattern)
        query = query.format(clauses=" OR ".join(note_clauses))
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY created_time DESC LIMIT 12"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _contact_rows_for_entity(
        self,
        conn: sqlite3.Connection,
        term: str,
        owner_scope: str | None = None,
        resolution: EntityResolution | None = None,
    ) -> list[ContactRow]:
        rows: list[ContactRow] = []
        seen: set[str] = set()
        resolution = resolution or self._resolve_entity(conn, term, owner_scope=owner_scope)

        for match in self._entity_matches(conn, term, owner_scope=owner_scope, resolution=resolution):
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

        for note in self._recent_notes_for_entity(conn, term, owner_scope=owner_scope, resolution=resolution):
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

    def _entity_profile_rows(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> list[dict[str, Any]]:
        resolution = self._resolve_entity(conn, term, owner_scope=owner_scope)
        patterns = self._entity_patterns(resolution)
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
                city,
                state,
                address,
                website,
                giro,
                otros_datos,
                unit_count,
                unit_type,
                phase,
                last_activity_time,
                created_time,
                modified_time
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
                city,
                state,
                address,
                website,
                giro,
                otros_datos,
                unit_count,
                unit_type,
                phase,
                last_activity_time,
                created_time,
                modified_time
            FROM contacts
        )
        WHERE (
            {clauses}
        )
        """
        profile_clauses = []
        params: list[Any] = []
        for pattern in patterns:
            for field in ["company_name", "contact_name", "full_name", "email", "phone"]:
                profile_clauses.append(f"lower(coalesce({field}, '')) LIKE lower(?)")
                params.append(pattern)
        query = query.format(clauses=" OR ".join(profile_clauses))
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY last_activity_time DESC, modified_time DESC LIMIT 8"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _entity_pending_tasks(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> list[dict[str, Any]]:
        resolution = self._resolve_entity(conn, term, owner_scope=owner_scope)
        patterns = self._entity_patterns(resolution)
        query = """
        SELECT owner_name, contact_name, subject, status, priority, due_date, description
        FROM tasks
        WHERE (status IS NULL OR lower(status) NOT IN ('completed', 'cancelled'))
          AND (
              {clauses}
          )
        """
        task_clauses = []
        params: list[Any] = []
        for pattern in patterns:
            for field in ["contact_name", "subject", "description"]:
                task_clauses.append(f"lower(coalesce({field}, '')) LIKE lower(?)")
                params.append(pattern)
        query = query.format(clauses=" OR ".join(task_clauses))
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY due_date ASC, owner_name LIMIT 8"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _recent_notes_by_owner(self, conn: sqlite3.Connection, owner_scope: str | None, limit: int = 6) -> list[dict[str, Any]]:
        if not owner_scope:
            return []
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT parent_name, created_time, title, content_text
                FROM notes
                WHERE owner_name = ?
                ORDER BY created_time DESC
                LIMIT ?
                """,
                [owner_scope, limit],
            ).fetchall()
        ]

    def _entity_brief(
        self,
        conn: sqlite3.Connection,
        term: str,
        owner_scope: str | None = None,
        question: str | None = None,
    ) -> dict[str, Any]:
        profiles = self._entity_profile_rows(conn, term, owner_scope=owner_scope)
        contacts = [row.__dict__ for row in self._contact_rows_for_entity(conn, term, owner_scope=owner_scope)]
        interactions = self._recent_interactions_for_entity(conn, term, owner_scope=owner_scope)
        notes = self._recent_notes_for_entity(conn, term, owner_scope=owner_scope)
        pending = self._entity_pending_tasks(conn, term, owner_scope=owner_scope)
        risks = self._risk_profile(conn, term, owner_scope=owner_scope)
        docs = self._document_search(conn, question or term, term)

        owners = sorted({row.get("owner_name") for row in profiles if row.get("owner_name")})
        phases = sorted({row.get("phase") for row in profiles if row.get("phase")})
        unit_rows = [row for row in profiles if row.get("unit_count")]
        latest_row = profiles[0] if profiles else {}
        return {
            "entity_term": term,
            "profiles": profiles,
            "owners": owners,
            "phases": phases,
            "unit_rows": unit_rows,
            "latest_row": latest_row,
            "contacts": contacts[:8],
            "recent_interactions": interactions[:8],
            "recent_notes": notes[:8],
            "pending_tasks": pending[:6],
            "document_chunks": docs[:4],
            "risk_profile": risks,
            "insights": self._entity_insights(term, latest_row, notes, interactions, pending),
        }

    def _entity_insights(
        self,
        term: str,
        latest_row: dict[str, Any],
        notes: list[dict[str, Any]],
        interactions: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> dict[str, Any]:
        combined = " || ".join((note.get("content_text") or "") for note in notes).lower()
        signals: list[str] = []
        objections: list[str] = []

        if any(token in combined for token in ["interes", "interés", "demo", "reuni", "visita", "cotiz", "propuesta"]):
            signals.append("Hay actividad comercial real y conversacion activa en notas recientes.")
        if "whatsapp" in combined:
            signals.append("El canal de seguimiento por WhatsApp aparece como viable.")
        if any(token in combined for token in ["dueño", "dueno", "decide", "gerente", "coordinador"]):
            signals.append("Las notas sugieren que ya hay informacion sobre quien influye o decide.")

        if any(token in combined for token in ["no cree", "rechaz", "caro", "objec", "ya tienen", "inversion"]):
            objections.append("Hay objeciones o resistencia frente al cambio en notas recientes.")
        if "rebot" in combined:
            objections.append("Hay correos rebotados o datos de contacto por depurar.")

        if pending:
            next_step = f"Dar seguimiento al compromiso abierto: {pending[0].get('subject') or 'sin asunto'}."
        elif "demo" in combined:
            next_step = "Buscar cierre de fecha para demo o validacion operativa."
        elif "cotiz" in combined or "propuesta" in combined:
            next_step = "Dar seguimiento a cotizacion/propuesta y confirmar decision maker."
        elif "visita" in combined or "reuni" in combined:
            next_step = "Convertir la ultima conversacion en siguiente accion concreta: demo, propuesta o reunion de seguimiento."
        elif interactions:
            next_step = "Retomar el ultimo contacto y pedir siguiente paso claro con fecha."
        else:
            next_step = f"Profundizar contexto comercial de {term} con una nueva llamada o nota de descubrimiento."

        status_parts = []
        if latest_row.get("phase"):
            status_parts.append(f"fase {latest_row['phase']}")
        if interactions:
            status_parts.append(f"{len(interactions)} interacciones recientes")
        if notes:
            status_parts.append(f"{len(notes)} notas")
        status = ", ".join(status_parts) if status_parts else "sin suficiente evidencia estructurada"

        return {
            "status": status,
            "signals": signals[:3],
            "objections": objections[:3],
            "next_step": next_step,
        }

    def _sales_draft(self, question: str, brief: dict[str, Any]) -> dict[str, Any]:
        if not brief:
            return {}
        latest = brief.get("latest_row") or {}
        entity_label = (
            latest.get("company_name")
            or latest.get("contact_name")
            or latest.get("full_name")
            or brief.get("entity_term")
            or "la cuenta solicitada"
        )
        contacts = brief.get("contacts") or []
        notes = brief.get("recent_notes") or []
        docs = brief.get("document_chunks") or []
        insights = brief.get("insights") or {}

        preferred_contact = next(
            (row for row in contacts if row.get("email") and row.get("email") != "Sin correo"),
            contacts[0] if contacts else None,
        )
        recipient_name = preferred_contact.get("label") if preferred_contact else f"equipo de {entity_label}"
        recipient_email = preferred_contact.get("email") if preferred_contact else None

        note_text = " || ".join((note.get("content_text") or "") for note in notes).lower()
        value_points: list[str] = []
        unit_count = latest.get("unit_count")
        if unit_count and any(char.isdigit() for char in str(unit_count)):
            unit_suffix = f" {latest.get('unit_type')}" if latest.get("unit_type") else ""
            value_points.append(f"una propuesta alineada a una operacion de {unit_count} unidades{unit_suffix}")
        if any("whatsapp" in (note.get("content_text") or "").lower() for note in notes):
            value_points.append("seguimiento agil por WhatsApp para acelerar coordinacion")
        if docs:
            value_points.append("respaldo con informacion tecnica y operativa de soluciones Flotimatics")
        if not value_points:
            value_points.append("una propuesta enfocada en visibilidad operativa, seguimiento y control comercial")

        product_focus: list[str] = []
        if any(token in note_text for token in ["demo", "reuni", "visita", "seguimiento"]):
            product_focus.append("una demo aterrizada a su operacion")
        if any(token in note_text for token in ["vehiculo", "unidades", "flota", "rastreo", "gps"]):
            product_focus.append("rastreo y monitoreo de flota")
        if any(token in note_text for token in ["google", "ya tienen", "actual"]):
            product_focus.append("mejoras frente a su esquema actual de seguimiento")
        if not product_focus:
            product_focus.extend(["rastreo y monitoreo de flota", "control operativo y seguimiento comercial"])

        subject = f"Seguimiento {entity_label}: propuesta Flotimatics"
        if any(token in note_text for token in ["demo", "visita", "reuni"]):
            subject = f"Seguimiento a conversacion con {entity_label}"

        intro = f"Hola {recipient_name},"
        body_lines = [
            intro,
            "",
            f"Quiero retomar la conversacion con {entity_label} para proponerte una siguiente etapa clara con Flotimatics.",
            f"Con base en lo que ya vimos, te podemos ayudar con {', '.join(product_focus[:2])}.",
            f"La idea es presentarte {value_points[0]} y revisar contigo el siguiente paso que mas valor les daria.",
        ]
        if len(value_points) > 1:
            body_lines.append(f"Tambien podemos sumar {value_points[1]}.")
        if insights.get("signals"):
            signal = str(insights["signals"][0]).strip().rstrip(".")
            body_lines.append(f"Vemos buen contexto para avanzar porque {signal.lower()}.")
        if insights.get("next_step"):
            next_step = str(insights["next_step"]).rstrip(".")
            body_lines.append(f"Mi recomendacion es {next_step}.")
        body_lines.extend(
            [
                "",
                "Si te hace sentido, te propongo agendar una llamada o demo corta esta semana para revisar su operacion y aterrizar una propuesta puntual.",
                "",
                "Quedo atento.",
                "Saludos,",
                "Equipo comercial Flotimatics",
            ]
        )

        return {
            "entity_name": entity_label,
            "recipient_name": recipient_name,
            "recipient_email": recipient_email if recipient_email and recipient_email != "Sin correo" else None,
            "subject": subject,
            "body": "\n".join(body_lines).strip(),
            "contacts": contacts[:4],
            "insights": insights,
            "question": question,
        }

    def _entity_action_plan(self, brief: dict[str, Any]) -> dict[str, Any]:
        if not brief:
            return {}
        latest = brief.get("latest_row") or {}
        entity_label = (
            latest.get("company_name")
            or latest.get("contact_name")
            or latest.get("full_name")
            or brief.get("entity_term")
            or "la cuenta solicitada"
        )
        contacts = brief.get("contacts") or []
        insights = brief.get("insights") or {}
        notes = brief.get("recent_notes") or []
        pending = brief.get("pending_tasks") or []
        recent = brief.get("recent_interactions") or []

        contact_line = None
        if contacts:
            top = contacts[0]
            contact_line = f"{top.get('label') or 'Contacto principal'}"
            if top.get("email") and top.get("email") != "Sin correo":
                contact_line += f" ({top['email']})"

        today = insights.get("next_step") or "Retomar la cuenta con un siguiente paso claro."
        tomorrow = "Enviar seguimiento con propuesta de valor y confirmar decision maker."
        if pending:
            tomorrow = f"Empujar el compromiso abierto: {pending[0].get('subject') or 'seguimiento comercial'}."
        week = "Buscar cierre de fecha para demo, reunion o propuesta concreta."
        if notes:
            week = "Convertir las notas recientes en una secuencia concreta: contacto, demo/propuesta y cierre de siguiente compromiso."

        return {
            "scope": entity_label,
            "headline": f"Plan comercial para {entity_label}",
            "summary": insights.get("status"),
            "contact_line": contact_line,
            "today": today,
            "tomorrow": tomorrow,
            "week": week,
            "signals": insights.get("signals") or [],
            "objections": insights.get("objections") or [],
            "recent_count": len(recent),
        }

    def _owner_action_plan(self, brief: dict[str, Any], owner_scope: str | None) -> dict[str, Any]:
        if not brief:
            return {}
        owner_name = brief.get("owner_name") or owner_scope or "el vendedor"
        priorities = brief.get("priorities") or []
        stale = brief.get("stale_contacts") or []
        pending = brief.get("pending_tasks") or []
        recent = brief.get("recent_activity") or []

        today_actions: list[str] = []
        for row in priorities[:3]:
            reasons = " / ".join(row.get("reasons") or [])
            today_actions.append(
                f"{row.get('related_name') or 'Sin nombre'}: {reasons or 'dar seguimiento prioritario hoy'}"
            )
        if not today_actions and recent:
            for row in recent[:3]:
                today_actions.append(
                    f"{row.get('related_name') or 'Sin nombre'}: retomar la conversacion reciente y cerrar siguiente paso."
                )
        if not today_actions:
            today_actions.append("Revisar cartera activa y recuperar oportunidades con señales comerciales recientes.")

        risks = []
        for row in stale[:3]:
            risks.append(f"{row.get('related_name') or 'Sin nombre'} lleva tiempo sin contacto.")
        if pending:
            risks.append(f"Tienes {len(pending)} compromiso(s) pendiente(s) por empujar.")

        return {
            "scope": owner_name,
            "headline": f"Plan comercial de hoy para {owner_name}",
            "summary": f"Hay {len(priorities)} seguimientos priorizados, {len(pending)} compromisos pendientes y {len(recent)} movimientos recientes visibles.",
            "today_actions": today_actions,
            "tomorrow": "Dar seguimiento a quienes respondan hoy y convertir interes en demo, reunion o propuesta.",
            "week": "Concentrar la semana en las cuentas mas calientes y rescatar 1 o 2 clientes frios recuperables.",
            "risks": risks,
        }

    def _sales_material(self, question: str, brief: dict[str, Any]) -> dict[str, Any]:
        if not brief:
            return {}
        latest = brief.get("latest_row") or {}
        entity_label = (
            latest.get("company_name")
            or latest.get("contact_name")
            or latest.get("full_name")
            or brief.get("entity_term")
            or "la cuenta solicitada"
        )
        insights = brief.get("insights") or {}
        contacts = brief.get("contacts") or []
        notes_text = " || ".join((row.get("content_text") or "") for row in (brief.get("recent_notes") or [])).lower()

        bullets = [
            "visibilidad operativa de la flota y seguimiento puntual",
            "mejor control de seguimiento comercial y compromisos",
            "acompanamiento para aterrizar una demo o siguiente paso claro",
        ]
        if "gps" in notes_text or "rastreo" in notes_text:
            bullets.insert(0, "mejoras sobre su esquema actual de rastreo y monitoreo")
        if "whatsapp" in notes_text:
            bullets.append("seguimiento agil por WhatsApp para acelerar coordinacion")

        objections = insights.get("objections") or []
        responses = []
        for objection in objections[:3]:
            responses.append(f"- Objecion: {objection}\n  Respuesta sugerida: bajar la friccion con demo corta, caso de uso concreto y siguiente compromiso con fecha.")

        return {
            "entity_name": entity_label,
            "question": question,
            "bullets": bullets[:5],
            "contact_name": contacts[0].get("label") if contacts else None,
            "next_step": insights.get("next_step"),
            "responses": responses,
        }

    def _entity_kpis(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> dict[str, Any]:
        profiles = self._entity_profile_rows(conn, term, owner_scope=owner_scope)
        interactions = self._recent_interactions_for_entity(conn, term, owner_scope=owner_scope)
        notes = self._recent_notes_for_entity(conn, term, owner_scope=owner_scope)
        pending = self._entity_pending_tasks(conn, term, owner_scope=owner_scope)
        owners = sorted({row.get("owner_name") for row in profiles if row.get("owner_name")})
        latest_touch = interactions[0]["interaction_at"] if interactions else None
        latest_name = None
        if profiles:
            latest_name = profiles[0].get("company_name") or profiles[0].get("contact_name") or profiles[0].get("full_name")
        return {
            "entity_term": term,
            "entity_name": latest_name or term,
            "owners": owners,
            "profile_matches": len(profiles),
            "interactions": len(interactions),
            "notes": len(notes),
            "pending_tasks": len(pending),
            "latest_touch": latest_touch,
            "unit_rows": [row for row in profiles if row.get("unit_count")],
        }

    def _entity_count_summary(self, conn: sqlite3.Connection, term: str, owner_scope: str | None = None) -> dict[str, Any]:
        profiles = self._entity_profile_rows(conn, term, owner_scope=owner_scope)
        interactions = self._recent_interactions_for_entity(conn, term, owner_scope=owner_scope)
        notes = self._recent_notes_for_entity(conn, term, owner_scope=owner_scope)
        leads = sum(1 for row in profiles if row.get("entity_type") == "lead")
        contacts = sum(1 for row in profiles if row.get("entity_type") == "contact")
        canonical_name = profiles[0].get("company_name") if profiles else term
        return {
            "entity_name": canonical_name or term,
            "total_profiles": len(profiles),
            "lead_count": leads,
            "contact_count": contacts,
            "interaction_count": len(interactions),
            "note_count": len(notes),
        }

    def _owner_client_count_summary(self, conn: sqlite3.Connection, owner_scope: str) -> dict[str, Any]:
        rows = self._assigned_clients(conn, owner_scope)
        unique_labels: list[str] = []
        seen: set[str] = set()
        for row in rows:
            label = row.get("company_name") or row.get("contact_name") or row.get("full_name")
            if not label:
                continue
            key = self._normalize_search_text(label)
            if key in seen:
                continue
            seen.add(key)
            unique_labels.append(label)
        return {
            "owner_name": owner_scope,
            "total_accounts": len(unique_labels),
            "sample_accounts": unique_labels[:5],
        }

    def _extract_rank_limit(self, normalized_question: str) -> int:
        digit_match = re.search(r"\btop\s+(\d+)\b|\b(\d+)\s+(?:clientes|prospectos|cuentas)\b", normalized_question)
        if digit_match:
            value = next((group for group in digit_match.groups() if group), None)
            if value:
                return max(1, min(10, int(value)))
        text_numbers = {
            "uno": 1,
            "dos": 2,
            "tres": 3,
            "cuatro": 4,
            "cinco": 5,
            "seis": 6,
            "siete": 7,
            "ocho": 8,
            "nueve": 9,
            "diez": 10,
        }
        for word, value in text_numbers.items():
            if re.search(rf"\b{word}\b", normalized_question):
                return value
        return 3

    def _owner_ranked_accounts(self, conn: sqlite3.Connection, owner_scope: str, limit: int = 3) -> list[dict[str, Any]]:
        query = """
        SELECT
            related_name,
            COUNT(*) AS total_interactions,
            SUM(CASE WHEN source_type = 'note' THEN 1 ELSE 0 END) AS note_count,
            SUM(CASE WHEN source_type = 'call' THEN 1 ELSE 0 END) AS call_count,
            MAX(interaction_at) AS last_touch
        FROM interactions
        WHERE owner_name = ?
          AND related_name IS NOT NULL
          AND trim(related_name) <> ''
        GROUP BY related_name
        ORDER BY MAX(interaction_at) DESC
        LIMIT 60
        """
        rows = [dict(row) for row in conn.execute(query, (owner_scope,)).fetchall()]
        ranked: list[dict[str, Any]] = []
        for row in rows:
            related_name = row.get("related_name")
            if not related_name or self._normalize_search_text(related_name) in {"sin nombre", "sin contacto"}:
                continue
            pending = len(self._entity_pending_tasks(conn, related_name, owner_scope=owner_scope))
            brief = self._entity_brief(conn, related_name, owner_scope=owner_scope, question=f"resumen de {related_name}")
            signals = brief.get("insights", {}).get("signals") or []
            last_touch = row.get("last_touch")
            recency_bonus = 0
            if last_touch:
                try:
                    parsed_touch = datetime.fromisoformat(str(last_touch).replace("Z", "+00:00"))
                    if parsed_touch.tzinfo is not None:
                        parsed_touch = parsed_touch.replace(tzinfo=None)
                    days_old = max(0, (datetime.now() - parsed_touch).days)
                    recency_bonus = max(0, 12 - min(days_old, 12))
                except ValueError:
                    recency_bonus = 0
            score = (
                int(row.get("total_interactions") or 0) * 3
                + int(row.get("note_count") or 0) * 2
                + int(row.get("call_count") or 0) * 3
                + pending * 4
                + recency_bonus
                + len(signals) * 4
            )
            ranked.append(
                {
                    "entity_name": related_name,
                    "score": score,
                    "total_interactions": int(row.get("total_interactions") or 0),
                    "note_count": int(row.get("note_count") or 0),
                    "call_count": int(row.get("call_count") or 0),
                    "pending_tasks": pending,
                    "last_touch": last_touch,
                    "signals": signals[:2],
                    "next_step": brief.get("insights", {}).get("next_step"),
                }
            )
        ranked.sort(key=lambda item: (-item["score"], -(item["call_count"]), -(item["note_count"]), item["entity_name"]))
        return ranked[:limit]

    def _owner_brief(self, conn: sqlite3.Connection, owner_scope: str | None) -> dict[str, Any]:
        if not owner_scope:
            return {}
        kpis = self._owner_kpis(conn, owner_scope)
        weekly_kpis = self._owner_kpis(conn, owner_scope, window_days=7)
        assigned_clients = self._assigned_clients(conn, owner_scope)
        pending_tasks = self._pending_tasks(conn, owner_scope)
        stale_contacts = self._stale_contacts(conn, owner_scope)
        recent_activity = self._recent_activity_by_owner(conn, owner_scope)
        priorities = self.get_priority_followups(owner=owner_scope, limit=6)
        recent_notes = self._recent_notes_by_owner(conn, owner_scope)
        return {
            "owner_name": owner_scope,
            "kpis": kpis,
            "weekly_kpis": weekly_kpis,
            "assigned_clients": assigned_clients[:8],
            "assigned_clients_count": len(assigned_clients),
            "pending_tasks": pending_tasks[:6],
            "stale_contacts": stale_contacts[:6],
            "recent_activity": recent_activity[:6],
            "priorities": priorities[:6],
            "recent_notes": recent_notes[:5],
        }

    def _team_brief(self, conn: sqlite3.Connection) -> dict[str, Any]:
        global_kpis = self._global_kpis(conn)
        weekly_kpis = self._global_kpis(conn, window_days=7)
        owner_load = self._owner_load(conn)
        pending_tasks = self._pending_tasks(conn)
        stale_contacts = self._stale_contacts(conn)
        recent_activity = self._recent_activity_by_owner(conn, None)
        priorities = self.get_priority_followups(limit=8)
        return {
            "global_kpis": global_kpis,
            "weekly_kpis": weekly_kpis,
            "owner_load": owner_load[:10],
            "pending_tasks": pending_tasks[:6],
            "stale_contacts": stale_contacts[:6],
            "recent_activity": recent_activity[:6],
            "priorities": priorities[:6],
        }

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

    def _owner_presence_summary(self, conn: sqlite3.Connection, owner_scope: str | None) -> dict[str, Any]:
        if not owner_scope:
            return {}
        interactions = conn.execute(
            "SELECT COUNT(*) AS total FROM interactions WHERE owner_name = ?",
            (owner_scope,),
        ).fetchone()
        notes = conn.execute(
            "SELECT COUNT(*) AS total FROM notes WHERE owner_name = ?",
            (owner_scope,),
        ).fetchone()
        tasks = conn.execute(
            "SELECT COUNT(*) AS total FROM tasks WHERE owner_name = ?",
            (owner_scope,),
        ).fetchone()
        recent = conn.execute(
            """
            SELECT related_name, interaction_at, source_type
            FROM interactions
            WHERE owner_name = ?
            ORDER BY interaction_at DESC
            LIMIT 5
            """,
            (owner_scope,),
        ).fetchall()
        mention_notes = conn.execute(
            """
            SELECT parent_name, created_time, owner_name, content_text
            FROM notes
            WHERE lower(coalesce(content_text,'')) LIKE lower(?)
            ORDER BY created_time DESC
            LIMIT 5
            """,
            (f"%{owner_scope}%",),
        ).fetchall()
        return {
            "owner_name": owner_scope,
            "interaction_count": interactions["total"] if interactions else 0,
            "note_count": notes["total"] if notes else 0,
            "task_count": tasks["total"] if tasks else 0,
            "recent_activity": [dict(row) for row in recent],
            "mentions_in_notes": [dict(row) for row in mention_notes],
        }

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

    def _latest_note(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> dict[str, Any] | None:
        query = """
        SELECT parent_name, owner_name, created_time, title, content_text
        FROM notes
        WHERE 1 = 1
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY created_time DESC LIMIT 1"
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

    def _overdue_tasks(self, conn: sqlite3.Connection, owner_scope: str | None = None) -> list[dict[str, Any]]:
        query = """
        SELECT owner_name, contact_name, subject, status, priority, due_date
        FROM tasks
        WHERE (status IS NULL OR lower(status) NOT IN ('completed', 'cancelled'))
          AND due_date IS NOT NULL
          AND date(due_date) < date('now')
        """
        params: list[Any] = []
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY due_date DESC LIMIT 10"
        return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _scheduled_calls(self, conn: sqlite3.Connection, owner_scope: str | None = None, days_ahead: int = 7) -> list[dict[str, Any]]:
        query = """
        SELECT related_name, owner_name, source_type, interaction_at, status, summary
        FROM interactions
        WHERE source_type = 'call'
          AND date(interaction_at) >= date('now')
          AND date(interaction_at) <= date('now', ?)
        """
        params: list[Any] = [f"+{days_ahead} day"]
        if owner_scope:
            query += " AND owner_name = ?"
            params.append(owner_scope)
        query += " ORDER BY interaction_at ASC LIMIT 12"
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
        rows = [dict(row) for row in conn.execute(query, params).fetchall()]
        for row in rows:
            row["source_kind"] = "task"
        if rows:
            return rows

        scheduled_calls = self._scheduled_calls(conn, owner_scope, days_ahead=2)
        for row in scheduled_calls:
            rows.append(
                {
                    "owner_name": row.get("owner_name"),
                    "contact_name": row.get("related_name"),
                    "subject": row.get("summary") or "Llamada programada",
                    "status": row.get("status"),
                    "priority": "programada",
                    "due_date": row.get("interaction_at"),
                    "source_kind": "call",
                }
            )
        overdue = self._overdue_tasks(conn, owner_scope)
        for row in overdue[:3]:
            item = dict(row)
            item["source_kind"] = "overdue_task"
            rows.append(item)
        return rows[:20]

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

    def _owner_comparison(self, conn: sqlite3.Connection, owner_names: list[str]) -> list[dict[str, Any]]:
        results = []
        for owner_name in owner_names:
            kpis = self._owner_kpis(conn, owner_name)
            assigned_clients = self._assigned_clients(conn, owner_name)
            pending_tasks = self._pending_tasks(conn, owner_name)
            note_count = next((row["total"] for row in kpis.get("by_type", []) if row["source_type"] == "note"), 0)
            call_count = next((row["total"] for row in kpis.get("by_type", []) if row["source_type"] == "call"), 0)
            task_count = next((row["total"] for row in kpis.get("by_type", []) if row["source_type"] == "task"), 0)
            results.append(
                {
                    "owner_name": owner_name,
                    "total_interactions": kpis.get("total_interactions", 0),
                    "unique_accounts": kpis.get("unique_accounts", 0),
                    "assigned_clients": len(assigned_clients),
                    "pending_tasks": len(pending_tasks),
                    "note_count": note_count,
                    "call_count": call_count,
                    "task_count": task_count,
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
            try:
                rows = conn.execute("SELECT DISTINCT name FROM owners WHERE name IS NOT NULL").fetchall()
            except sqlite3.OperationalError:
                rows = []
            for row in rows:
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
                " mio ",
                " mia ",
                " mios ",
                " mias ",
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
                " dinero en la mesa ",
                " oportunidades ",
                " si solo pudiera hacer una accion hoy ",
                " que tres clientes debo atacar primero esta semana ",
                " conclusion y luego evidencia sobre mi cartera ",
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
            "kpi mio",
            "mis kpis",
            "mis contactos",
            "mis leads",
            "mis clientes",
            "mis prospectos",
            "mi cartera",
            "clientes calientes",
            "clientes frios",
            "analiza mis notas",
            "que oportunidades tengo",
            "dinero en la mesa",
            "si solo pudiera hacer una accion hoy",
            "que tres clientes debo atacar primero esta semana",
            "dame primero conclusion y luego evidencia sobre mi cartera",
        ]
        if any(self._normalize_search_text(signal) in normalized for signal in seller_default_signals):
            return user.crm_owner_name
        return None

    def _normalize_user_scoped_question(self, question: str, owner_scope: str | None) -> str:
        if not owner_scope:
            return question
        normalized = self._normalize_search_text(question)
        if any(token in f" {normalized} " for token in [" mi ", " mis ", " mio ", " mia ", " mios ", " mias ", " yo ", " conmigo "]):
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
            "title": user.title,
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
            "de alta",
            "cliente ",
            "prospecto ",
            "contacto ",
            "empresa ",
            "seguimiento para ",
            "seguimiento a ",
            "venta para una llamada con ",
            "una llamada con ",
            "llamada con ",
            "la principal para ",
            "el principal para ",
            "el mejor para ",
            "la mejor para ",
        ]
        removable_suffixes = [
            ", dame solamente eso",
            " dame solamente eso",
            ", solamente eso",
            " solamente eso",
            ", solo eso",
            " solo eso",
            ", nada mas",
            " nada mas",
            ", nada más",
            " nada mas dame eso",
        ]
        changed = True
        while changed:
            changed = False
            for prefix in removable_prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    changed = True
            for suffix in removable_suffixes:
                if cleaned.endswith(suffix):
                    cleaned = cleaned[: -len(suffix)].strip(" ,.;:-")
                    changed = True
        connector_patterns = [
            r"\s+y luego\s+.+$",
            r"\s+despues\s+.+$",
            r"\s+después\s+.+$",
            r"\s+y redacta\s+.+$",
            r"\s+y escribe\s+.+$",
            r"\s+y arma\s+.+$",
            r"\s+y hazme\s+.+$",
            r"\s+y fabrica\s+.+$",
            r"\s+y proponme\s+.+$",
        ]
        for pattern in connector_patterns:
            cleaned = re.sub(pattern, "", cleaned).strip(" ,.;:-")
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

    def _format_entity_brief(self, brief: dict[str, Any]) -> str:
        if not brief:
            return "No encontre evidencia suficiente para armar un resumen comercial."
        latest = brief.get("latest_row") or {}
        entity_label = (
            latest.get("company_name")
            or latest.get("contact_name")
            or latest.get("full_name")
            or brief.get("entity_term")
            or "la cuenta solicitada"
        )
        lines = [f"Resumen comercial de {entity_label}"]
        insights = brief.get("insights") or {}
        if insights.get("status"):
            lines.append(f"Lectura actual: {insights['status']}")
        if insights.get("signals"):
            lines.append("Señales comerciales: " + " ".join(insights["signals"]))
        if insights.get("objections"):
            lines.append("Objeciones o fricciones: " + " ".join(insights["objections"]))
        if insights.get("next_step"):
            lines.append(f"Siguiente paso sugerido: {insights['next_step']}")
        owners = brief.get("owners") or []
        phases = brief.get("phases") or []
        if owners:
            lines.append(f"Responsables detectados: {', '.join(owners)}")
        if phases:
            lines.append(f"Fase comercial: {', '.join(phases)}")
        if latest.get("giro"):
            lines.append(f"Giro: {latest['giro']}")
        if latest.get("unit_count"):
            unit_suffix = f" ({latest.get('unit_type')})" if latest.get("unit_type") else ""
            lines.append(f"Unidades reportadas: {latest['unit_count']}{unit_suffix}")
        location = ", ".join(part for part in [latest.get("city"), latest.get("state")] if part)
        if location:
            lines.append(f"Ubicacion: {location}")
        if latest.get("website"):
            lines.append(f"Sitio: {latest['website']}")

        contacts = brief.get("contacts") or []
        if contacts:
            lines.append("")
            lines.append("Contactos y datos detectados:")
            for row in contacts[:5]:
                lines.append(
                    f"- {row.get('label') or 'Sin nombre'} | {row.get('email') or 'Sin correo'} | {row.get('phone') or 'Sin telefono'}"
                )

        interactions = brief.get("recent_interactions") or []
        if interactions:
            lines.append("")
            lines.append("Interacciones recientes:")
            for row in interactions[:5]:
                summary = row.get("summary") or row.get("status") or "-"
                lines.append(
                    f"- {row.get('interaction_at') or '-'} | {row.get('source_type') or '-'} | {summary[:140]}"
                )

        pending = brief.get("pending_tasks") or []
        if pending:
            lines.append("")
            lines.append("Compromisos abiertos:")
            for row in pending[:4]:
                lines.append(
                    f"- {row.get('due_date') or 'Sin fecha'} | {row.get('subject') or 'Sin asunto'} | {row.get('status') or 'Sin estatus'}"
                )

        notes = brief.get("recent_notes") or []
        if notes:
            lines.append("")
            lines.append("Notas clave:")
            for row in notes[:4]:
                snippet = (row.get("content_text") or "").replace("\n", " ").strip()
                lines.append(f"- {row.get('created_time') or '-'} | {snippet[:180]}")

        docs = brief.get("document_chunks") or []
        if docs:
            lines.append("")
            lines.append("Soporte en PDFs:")
            for chunk in docs[:3]:
                snippet = (chunk.get("content") or "").replace("\n", " ").strip()
                lines.append(f"- {chunk.get('file_name') or '-'} | {snippet[:170]}")

        risks = (brief.get("risk_profile") or {}).get("risks") or []
        if risks:
            lines.append("")
            lines.append("Alertas:")
            for risk in risks[:3]:
                lines.append(f"- {risk}")
        return "\n".join(lines)

    def _format_sales_draft(self, draft: dict[str, Any]) -> str:
        if not draft:
            return "No encontre suficiente contexto para redactar un correo comercial util."
        lines = [f"Borrador de correo para {draft.get('entity_name') or 'la cuenta solicitada'}"]
        if draft.get("recipient_email"):
            lines.append(f"Para sugerido: {draft['recipient_email']}")
        elif draft.get("recipient_name"):
            lines.append(f"Contacto sugerido: {draft['recipient_name']}")
        lines.append(f"Asunto: {draft.get('subject') or 'Seguimiento comercial Flotimatics'}")
        lines.append("")
        lines.append(draft.get("body") or "")

        contacts = draft.get("contacts") or []
        if contacts:
            lines.append("")
            lines.append("Contactos disponibles para enviarlo:")
            for row in contacts[:4]:
                lines.append(
                    f"- {row.get('label') or 'Sin nombre'} | {row.get('email') or 'Sin correo'} | {row.get('phone') or 'Sin telefono'}"
                )
        return "\n".join(lines)

    def _format_action_plan(self, plan: dict[str, Any]) -> str:
        if not plan:
            return "No encontre suficiente contexto para proponer un plan accionable."
        lines = [plan.get("headline") or "Plan de accion"]
        if plan.get("summary"):
            lines.append(plan["summary"])
        if plan.get("contact_line"):
            lines.append(f"Contacto recomendado: {plan['contact_line']}")

        today_actions = plan.get("today_actions")
        if today_actions:
            lines.append("")
            lines.append("Prioridades de hoy:")
            for item in today_actions[:4]:
                lines.append(f"- {item}")
        elif plan.get("today"):
            lines.append("")
            lines.append(f"Hoy: {plan['today']}")

        if plan.get("tomorrow"):
            lines.append(f"Mañana: {plan['tomorrow']}")
        if plan.get("week"):
            lines.append(f"Esta semana: {plan['week']}")

        risks = plan.get("risks") or []
        objections = plan.get("objections") or []
        if risks or objections:
            lines.append("")
            lines.append("Riesgos a cuidar:")
            for risk in (risks + objections)[:4]:
                lines.append(f"- {risk}")
        return "\n".join(lines)

    def _format_sales_material(self, material: dict[str, Any]) -> str:
        if not material:
            return "No encontre suficiente contexto para preparar ese material comercial."
        question = self._normalize_search_text(material.get("question") or "")
        entity_name = material.get("entity_name") or "la cuenta solicitada"
        bullets = material.get("bullets") or []
        next_step = material.get("next_step")

        if "whatsapp" in question:
            opening = material.get("contact_name") or "equipo"
            lines = [
                f"WhatsApp sugerido para {entity_name}:",
                f"Hola {opening}, te escribo para retomar la conversacion con {entity_name}.",
                f"Creo que Flotimatics puede ayudarte especialmente con {bullets[0] if bullets else 'visibilidad operativa y seguimiento comercial'}.",
                "Si te hace sentido, armamos una llamada o demo corta esta semana.",
            ]
            return "\n".join(lines)

        if "speech" in question:
            return (
                f"Speech de 30 segundos para {entity_name}:\n"
                f"\"En Flotimatics les podemos ayudar con {', '.join(bullets[:2])}. "
                f"La idea no es solo mostrar tecnologia, sino aterrizar una solucion que les deje mas control operativo y un siguiente paso claro. "
                f"{next_step or 'Si hace sentido, propondria una demo corta para aterrizarlo.'}\""
            )

        if "agenda" in question:
            lines = [f"Mini agenda de reunion para {entity_name}:"]
            lines.extend(
                [
                    "- Repaso rapido del contexto actual del cliente",
                    "- Deteccion de necesidad operativa y dolor principal",
                    "- Presentacion de valor Flotimatics",
                    f"- {next_step or 'Definicion de siguiente paso con fecha'}",
                ]
            )
            return "\n".join(lines)

        if "preguntas de descubrimiento" in question:
            return "\n".join(
                [
                    f"Preguntas de descubrimiento para {entity_name}:",
                    "- ¿Como estan resolviendo hoy el seguimiento operativo y comercial?",
                    "- ¿Que les esta costando mas trabajo en la operacion actual?",
                    "- ¿Quien participa en la decision y que necesita ver para avanzar?",
                    "- ¿Que resultado tendria que lograr una demo o propuesta para que valga la pena?",
                ]
            )

        if "objeciones probables" in question or "como responderlas" in question:
            responses = material.get("responses") or []
            if responses:
                return "\n".join([f"Objeciones probables para {entity_name}:"] + responses)
            return "\n".join(
                [
                    f"Objeciones probables para {entity_name}:",
                    "- Objecion: ya tienen una solucion actual\n  Respuesta sugerida: comparar contra dolores actuales y proponer demo enfocada a mejora concreta.",
                    "- Objecion: no ven urgencia\n  Respuesta sugerida: aterrizar costo de seguir igual y siguiente paso pequeno, no invasivo.",
                ]
            )

        lines = [f"Material comercial para {entity_name}:"]
        if "argumentos de venta" in question:
            lines[0] = f"Argumentos de venta para {entity_name}:"
        elif "propuesta comercial" in question:
            lines[0] = f"Propuesta comercial breve para {entity_name}:"
        elif "beneficios" in question:
            lines[0] = f"Beneficios recomendados para {entity_name}:"
        elif "bullets de valor" in question:
            lines[0] = f"Bullets de valor para {entity_name}:"
        for bullet in bullets[:5]:
            lines.append(f"- {bullet}")
        if next_step:
            lines.append(f"Siguiente paso sugerido: {next_step}")
        return "\n".join(lines)

    def _format_owner_brief(self, brief: dict[str, Any], owner_scope: str | None) -> str:
        if not brief:
            owner_label = owner_scope or "el vendedor solicitado"
            return f"No encontre evidencia suficiente para resumir la cartera de {owner_label}."
        owner_name = brief.get("owner_name") or owner_scope or "el vendedor"
        kpis = brief.get("kpis") or {}
        weekly_kpis = brief.get("weekly_kpis") or {}
        lines = [
            f"Panorama comercial de {owner_name}",
            f"Interacciones totales: {kpis.get('total_interactions', 0)}",
            f"Cuentas unicas activas: {kpis.get('unique_accounts', 0)}",
            f"Clientes y prospectos asignados: {brief.get('assigned_clients_count', 0)}",
            f"Interacciones ultimos 7 dias: {weekly_kpis.get('total_interactions', 0)}",
        ]
        by_type = kpis.get("by_type") or []
        if by_type:
            lines.append("Mix de actividad: " + ", ".join(f"{row['source_type']} {row['total']}" for row in by_type[:4]))

        priorities = brief.get("priorities") or []
        if priorities:
            lines.append("")
            lines.append("Clientes calientes o de mayor prioridad:")
            for row in priorities[:4]:
                reasons = " / ".join(row.get("reasons") or [])
                lines.append(
                    f"- {row.get('related_name') or 'Sin nombre'} | prioridad {row.get('priority_label') or '-'} | {reasons[:160]}"
                )

        stale = brief.get("stale_contacts") or []
        if stale:
            lines.append("")
            lines.append("Clientes frios o rezagados:")
            for row in stale[:4]:
                lines.append(f"- {row.get('related_name') or 'Sin nombre'} | ultimo toque {row.get('last_touch') or '-'}")

        pending = brief.get("pending_tasks") or []
        if pending:
            lines.append("")
            lines.append("Compromisos pendientes:")
            for row in pending[:4]:
                lines.append(
                    f"- {row.get('due_date') or 'Sin fecha'} | {row.get('contact_name') or 'Sin contacto'} | {row.get('subject') or 'Sin asunto'}"
                )

        recent_activity = brief.get("recent_activity") or []
        if recent_activity:
            lines.append("")
            lines.append("Actividad reciente:")
            for row in recent_activity[:4]:
                lines.append(
                    f"- {row.get('interaction_at') or '-'} | {row.get('related_name') or 'Sin nombre'} | {row.get('source_type') or '-'}"
                )
        return "\n".join(lines)

    def _format_team_brief(self, brief: dict[str, Any]) -> str:
        if not brief:
            return "No encontre evidencia suficiente para resumir al equipo comercial."
        global_kpis = brief.get("global_kpis") or {}
        weekly_kpis = brief.get("weekly_kpis") or {}
        lines = [
            "Panorama comercial del equipo Flotimatics",
            f"Interacciones totales: {global_kpis.get('total_interactions', 0)}",
            f"Cuentas unicas: {global_kpis.get('unique_accounts', 0)}",
            f"Vendedores con actividad: {global_kpis.get('owners', 0)}",
            f"Interacciones ultimos 7 dias: {weekly_kpis.get('total_interactions', 0)}",
        ]
        by_owner = global_kpis.get("by_owner") or []
        if by_owner:
            lines.append("")
            lines.append("Lideres por actividad:")
            for row in by_owner[:5]:
                lines.append(f"- {row['owner_name']}: {row['total']} interacciones")

        owner_load = brief.get("owner_load") or []
        if owner_load:
            lines.append("")
            lines.append("Carga por vendedor:")
            for row in owner_load[:5]:
                lines.append(f"- {row['owner_name']}: {row['total_records']} cuentas")

        priorities = brief.get("priorities") or []
        if priorities:
            lines.append("")
            lines.append("Seguimientos prioritarios del equipo:")
            for row in priorities[:4]:
                lines.append(
                    f"- {row.get('related_name') or 'Sin nombre'} | {row.get('owner_name') or 'Sin responsable'} | prioridad {row.get('priority_label') or '-'}"
                )
        return "\n".join(lines)

    def _format_entity_kpis(self, kpis: dict[str, Any], window_days: int | None) -> str:
        if not kpis:
            return "No encontre indicadores del cliente o prospecto solicitado."
        entity_name = kpis.get("entity_name") or kpis.get("entity_term") or "la cuenta solicitada"
        header = f"Indicadores comerciales de {entity_name}"
        if window_days:
            header += f" en los ultimos {window_days} dias"
        lines = [
            header,
            f"Coincidencias CRM: {kpis.get('profile_matches', 0)}",
            f"Interacciones recientes detectadas: {kpis.get('interactions', 0)}",
            f"Notas CRM detectadas: {kpis.get('notes', 0)}",
            f"Compromisos abiertos: {kpis.get('pending_tasks', 0)}",
        ]
        if kpis.get("owners"):
            lines.append(f"Responsables CRM: {', '.join(kpis['owners'])}")
        if kpis.get("latest_touch"):
            lines.append(f"Ultimo movimiento detectado: {kpis['latest_touch']}")
        unit_rows = kpis.get("unit_rows") or []
        if unit_rows:
            row = unit_rows[0]
            unit_suffix = f" ({row.get('unit_type')})" if row.get("unit_type") else ""
            lines.append(f"Unidades reportadas: {row.get('unit_count')}{unit_suffix}")
        return "\n".join(lines)

    def _format_entity_suggestions(self, term: str, suggestions: list[str]) -> str:
        if not suggestions:
            return f'No encontre informacion registrada para "{term}".'
        lines = [f'No encontre informacion registrada exactamente para "{term}".']
        lines.append("Lo mas cercano que si veo en CRM es:")
        lines.extend(f"- {name}" for name in suggestions[:3])
        lines.append("Si quieres, prueba con uno de esos nombres y te doy el resumen, plan o contactos.")
        return "\n".join(lines)

    def _format_entity_count_summary(self, summary: dict[str, Any]) -> str:
        if not summary:
            return "No encontre registros para esa cuenta."
        return "\n".join(
            [
                f"Registros detectados para {summary.get('entity_name') or 'la cuenta solicitada'}:",
                f"- Fichas estructuradas en CRM: {summary.get('total_profiles', 0)}",
                f"- Leads: {summary.get('lead_count', 0)}",
                f"- Contacts: {summary.get('contact_count', 0)}",
                f"- Interacciones recientes detectadas: {summary.get('interaction_count', 0)}",
                f"- Notas CRM detectadas: {summary.get('note_count', 0)}",
            ]
        )

    def _format_owner_client_count_summary(self, summary: dict[str, Any], owner_scope: str | None) -> str:
        if not summary:
            owner_label = owner_scope or "ese vendedor"
            return f"No encontre clientes o prospectos visibles para {owner_label}."
        lines = [
            f"{summary.get('owner_name') or owner_scope or 'Ese vendedor'} tiene {summary.get('total_accounts', 0)} clientes o prospectos visibles en leads/contacts."
        ]
        sample_accounts = summary.get("sample_accounts") or []
        if sample_accounts:
            lines.append("Ejemplos:")
            lines.extend(f"- {label}" for label in sample_accounts[:5])
        return "\n".join(lines)

    def _format_owner_load(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre propietarios con registros."
        return "\n".join(f"{row['owner_name']}: {row['total_records']}" for row in rows)

    def _format_owner_ranked_accounts(self, rows: list[dict[str, Any]], owner_scope: str | None, limit: int) -> str:
        if not rows:
            owner_label = owner_scope or "ese vendedor"
            return f"No encontre cuentas suficientes para rankear a {owner_label}."
        owner_label = owner_scope or "ese vendedor"
        lines = [f"Top {min(limit, len(rows))} clientes o prospectos de {owner_label}:"]
        for index, row in enumerate(rows[:limit], start=1):
            reasons = []
            if row.get("total_interactions"):
                reasons.append(f"{row['total_interactions']} interacciones")
            if row.get("note_count"):
                reasons.append(f"{row['note_count']} notas")
            if row.get("call_count"):
                reasons.append(f"{row['call_count']} llamadas")
            if row.get("pending_tasks"):
                reasons.append(f"{row['pending_tasks']} pendientes")
            signals = row.get("signals") or []
            if signals:
                reasons.append(signals[0])
            reason_text = ", ".join(reasons) if reasons else "actividad comercial visible"
            line = f"{index}. {row['entity_name']} | {reason_text}"
            if row.get("next_step"):
                line += f" | Siguiente paso: {row['next_step']}"
            lines.append(line)
        return "\n".join(lines)

    def _format_assigned_clients(self, rows: list[dict[str, Any]], owner_scope: str | None) -> str:
        if not rows:
            if owner_scope:
                with self._connect() as conn:
                    presence = self._owner_presence_summary(conn, owner_scope)
                if presence.get("interaction_count") or presence.get("note_count"):
                    lines = [
                        f"No veo clientes o prospectos asignados actualmente a {owner_scope} en leads/contacts.",
                        f"Pero si aparece como owner historico con {presence.get('interaction_count', 0)} interacciones y {presence.get('note_count', 0)} notas.",
                    ]
                    recent = presence.get("recent_activity") or []
                    if recent:
                        lines.append("Actividad historica visible:")
                        for row in recent[:3]:
                            lines.append(
                                f"- {row.get('interaction_at') or '-'} | {row.get('related_name') or 'Sin nombre'} | {row.get('source_type') or '-'}"
                            )
                    mentions = presence.get("mentions_in_notes") or []
                    if mentions:
                        lines.append("Tambien aparece mencionado en notas como contexto de cuentas:")
                        for row in mentions[:2]:
                            snippet = (row.get("content_text") or "").replace("\n", " ").strip()
                            lines.append(
                                f"- {row.get('parent_name') or 'Sin cliente'} | {row.get('created_time') or '-'} | {snippet[:140]}"
                            )
                    return "\n".join(lines)
                return (
                    f"No encontre clientes o prospectos asignados a {owner_scope}. "
                    "Puede significar que su cartera actual no tiene registros visibles en leads/contacts o que el nombre del owner en CRM es distinto."
                )
            return "No encontre clientes asignados."
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
            owner_label = owner_scope or "el equipo"
            return f"No encontre actividad reciente para {owner_label} en el rango consultado."
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
            return "No encontre compromisos pendientes para hoy. Si esperabas actividad, revisa tareas sin fecha, vencidas o registradas con otro owner."
        lines = []
        for row in rows:
            source_kind = row.get("source_kind") or "task"
            if source_kind == "call":
                prefix = "Llamada cercana"
            elif source_kind == "overdue_task":
                prefix = "Pendiente vencido"
            else:
                prefix = "Pendiente de hoy"
            lines.append(
                f"{prefix} | {row.get('owner_name') or 'Sin responsable'} | {row.get('contact_name') or 'Sin contacto'} | {row.get('subject') or 'Sin asunto'} | {row.get('priority') or 'Sin prioridad'} | {row.get('due_date') or 'Sin fecha'}"
            )
        return "\n".join(lines)

    def _format_stale_contacts(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre clientes con mas de 30 dias sin contacto."
        return "\n".join(
            f"{row.get('related_name') or 'Sin nombre'} | {row.get('owner_name') or 'Sin responsable'} | {row.get('last_touch') or '-'}"
            for row in rows
        )

    def _format_interaction_list(self, rows: list[dict[str, Any]], fallback: str) -> str:
        if not rows:
            if fallback:
                return fallback + " Si esperabas movimiento, revisa si la actividad se guardo como nota, llamada futura o bajo otro responsable."
            return "No encontre interacciones para ese criterio."
        return "\n".join(
            f"{row.get('related_name') or 'Sin nombre'} | {row.get('source_type') or '-'} | {row.get('interaction_at') or '-'} | {row.get('owner_name') or 'Sin responsable'}"
            for row in rows
        )

    def _format_yesterday_contacts(self, conn: sqlite3.Connection, rows: list[dict[str, Any]], owner_scope: str | None) -> str:
        if rows:
            return self._format_interaction_list(rows, "")
        recent = self._recent_activity_by_owner(conn, owner_scope)
        if recent:
            latest = recent[:3]
            lines = ["No veo contactos registrados ayer para ese owner. Lo mas reciente visible es:"]
            for row in latest:
                lines.append(
                    f"- {row.get('interaction_at') or '-'} | {row.get('related_name') or 'Sin nombre'} | {row.get('source_type') or '-'}"
                )
            return "\n".join(lines)
        return "No veo contactos registrados ayer ni actividad reciente visible para ese owner."

    def _format_latest_contacted(self, row: dict[str, Any] | None) -> str:
        if not row:
            return "No encontre un ultimo contacto registrado."
        return (
            f"{row.get('related_name') or 'Sin nombre'} | "
            f"{row.get('source_type') or '-'} | "
            f"{row.get('interaction_at') or '-'} | "
            f"{row.get('owner_name') or 'Sin responsable'}"
        )

    def _format_latest_note(self, row: dict[str, Any] | None, owner_scope: str | None) -> str:
        if not row:
            owner_label = owner_scope or "el sistema"
            return f"No encontre notas recientes para {owner_label}."
        snippet = (row.get("content_text") or "").replace("\n", " ").strip()
        return (
            f"Ultima nota registrada: {row.get('created_time') or '-'} | "
            f"{row.get('parent_name') or 'Sin cliente'} | "
            f"{row.get('owner_name') or 'Sin responsable'} | "
            f"{snippet[:220]}"
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

    def _format_owner_comparison(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No encontre evidencia suficiente para comparar vendedores."
        return "\n".join(
            f"{row['owner_name']} | Interacciones: {row['total_interactions']} | Cuentas unicas: {row['unique_accounts']} | Clientes asignados: {row['assigned_clients']} | Tareas pendientes: {row['pending_tasks']} | Notas: {row['note_count']} | Llamadas: {row['call_count']} | Tasks: {row['task_count']}"
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

    def _format_owner_comparison_analysis(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return (
                "Hechos:\n"
                "- No encontre evidencia suficiente para comparar vendedores.\n\n"
                "Interpretacion:\n"
                "- La base no ofrece metricas suficientes para contrastar su actividad.\n\n"
                "Recomendacion:\n"
                "- Verifica sincronizacion de Zoho y vuelve a intentar."
            )
        facts = [
            f"- {row['owner_name']}: {row['total_interactions']} interacciones, {row['unique_accounts']} cuentas unicas, {row['assigned_clients']} clientes asignados y {row['pending_tasks']} tareas pendientes."
            for row in rows
        ]
        best = max(rows, key=lambda item: (item["total_interactions"], item["unique_accounts"], item["assigned_clients"]))
        return "\n".join(
            [
                "Hechos:",
                *facts,
                "",
                "Interpretacion:",
                f"- {best['owner_name']} muestra la actividad comercial mas fuerte con la evidencia actual del CRM.",
                "",
                "Recomendacion:",
                f"- Usa a {best['owner_name']} como referencia de actividad y revisa donde el otro vendedor tiene menos movimiento o menos cartera activa.",
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
        elif evidence.get("action_plan"):
            sections.append(self._format_action_plan(evidence["action_plan"]))
        elif evidence.get("sales_material"):
            sections.append(self._format_sales_material(evidence["sales_material"]))
        elif evidence.get("sales_draft"):
            sections.append(self._format_sales_draft(evidence["sales_draft"]))
        elif evidence.get("entity_brief"):
            sections.append(self._format_entity_brief(evidence["entity_brief"]))
        elif evidence.get("owner_brief"):
            sections.append(self._format_owner_brief(evidence["owner_brief"], evidence.get("owner_scope")))
        elif evidence.get("team_brief"):
            sections.append(self._format_team_brief(evidence["team_brief"]))
        elif evidence.get("entity_suggestions") and evidence.get("entity_hint"):
            sections.append(self._format_entity_suggestions(evidence["entity_hint"], evidence["entity_suggestions"]))
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
            "Eres el asistente comercial del equipo Flotimatics.",
            "Ayudas a direccion y vendedores con KPIs, cartera, comparativas, historial de notas, compromisos, llamadas y contexto comercial por cliente.",
            "Responde en español claro, ejecutivo y accionable.",
            "Usa solo la evidencia proporcionada de Zoho CRM y PDFs locales.",
            "No inventes contactos, teléfonos, correos, responsables, unidades ni conclusiones.",
            "Si la evidencia es parcial, entrega lo que sí está respaldado y di claramente lo que falta.",
            "Cuando la pregunta sea amplia, sintetiza primero el panorama y luego baja a datos concretos.",
            "Si te piden propuesta, comentario comercial, formato, redaccion o plan, responde como asesor comercial y no solo como buscador de datos.",
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
            f"Resumen cliente: {evidence.get('entity_brief')}",
            f"Resumen vendedor: {evidence.get('owner_brief')}",
            f"Resumen equipo: {evidence.get('team_brief')}",
            f"Plan sugerido: {evidence.get('action_plan')}",
            f"Material comercial sugerido: {evidence.get('sales_material')}",
            f"Borrador comercial sugerido: {evidence.get('sales_draft')}",
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
