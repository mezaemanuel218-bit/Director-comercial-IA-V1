import unicodedata
from dataclasses import dataclass
from typing import Literal


ResponseMode = Literal["data", "analysis", "hybrid"]


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(ascii_text.lower().split())


@dataclass
class QuestionIntent:
    mode: ResponseMode
    wants_web: bool
    asks_for_emails: bool
    asks_for_names: bool
    asks_for_phones: bool
    asks_for_contact_directory: bool
    asks_for_owner_load: bool
    asks_for_assigned_clients: bool
    asks_for_interactions_by_owner: bool
    asks_for_generic_relative_interactions: bool
    asks_for_recent_activity_by_owner: bool
    asks_for_risks: bool
    asks_for_kpis: bool
    asks_for_global_kpis: bool
    asks_for_weekly_window: bool
    asks_for_last_contact: bool
    asks_for_pending_commitments: bool
    asks_for_stale_contacts: bool
    asks_for_today_call_list: bool
    asks_for_comparison: bool
    asks_for_yesterday_contacts: bool
    asks_for_day_before_yesterday_contacts: bool
    asks_for_latest_contacted: bool
    asks_for_today_pending: bool


def classify_question(question: str) -> QuestionIntent:
    q = _normalize(question)

    asks_for_emails = any(token in q for token in ["correo", "correos", "mail", "email", "emails"])
    asks_for_names = any(token in q for token in ["nombre", "nombres", "persona", "personas", "contacto", "contactos"])
    asks_for_phones = any(token in q for token in ["telefono", "telefonos", "numero", "numeros", "celular", "movil"])
    asks_for_contact_directory = any(token in q for token in ["contacto", "contactos"]) and any(token in q for token in [" de ", " del ", ":"])

    asks_for_owner_load = (
        (any(token in q for token in ["cuantos clientes", "asignados", "carga"]) and any(token in q for token in ["vendedor", "propietario", "agente", "clientes"]))
        or any(token in q for token in ["quien tiene mas clientes asignados", "carga por vendedor", "quien tiene mas prospectos asignados"])
    )

    asks_for_assigned_clients = any(
        token in q
        for token in [
            "mis clientes",
            "quienes son mis clientes",
            "clientes de ",
            "clientes asignados de ",
            "prospectos de ",
            "prospectos asignados de ",
        ]
    )

    asks_for_interactions_by_owner = "interacciones" in q and any(token in q for token in ["vendedor", "agente", "propietario", "cada"])
    asks_for_generic_relative_interactions = any(token in q for token in ["interacciones", "actividad"]) and any(token in q for token in ["ayer", "antier", "anteayer"])
    asks_for_recent_activity_by_owner = any(token in q for token in ["actividad reciente", "mas actividad reciente"]) and any(token in q for token in ["quien", "vendedor"])
    asks_for_risks = "riesgo" in q or "riesgos" in q
    asks_for_kpis = "kpi" in q or "kpis" in q or ("metric" in q and any(token in q for token in ["nota", "cliente"]))
    asks_for_global_kpis = asks_for_kpis and any(token in q for token in ["global", "todos", "vendedores", "equipo"])
    asks_for_weekly_window = "semana" in q
    asks_for_last_contact = any(token in q for token in ["ultimo contacto", "ayer", "antier", "anteayer"])
    asks_for_pending_commitments = any(token in q for token in ["compromiso", "pendiente", "tarea"])
    asks_for_stale_contacts = any(token in q for token in ["30 dias", "sin contacto"])
    asks_for_today_call_list = "a quien debo llamar hoy" in q or "hoy y por que" in q
    asks_for_comparison = any(token in q for token in ["compara", "comparativa", " vs ", "diferencia entre", "diferencias entre"])
    wants_web = any(token in q for token in ["web", "internet"])
    asks_for_yesterday_contacts = "ayer" in q and any(token in q for token in ["hable", "llame", "contacte"])
    asks_for_day_before_yesterday_contacts = any(token in q for token in ["antier", "anteayer"]) and any(token in q for token in ["hable", "llame", "contacte"])
    asks_for_latest_contacted = any(token in q for token in ["ultimo cliente", "cliente que se contacto a lo ultimo"])
    asks_for_today_pending = "hoy" in q and any(token in q for token in ["pendiente", "compromiso", "tarea"])

    analysis_signals = ["plan", "estrategia", "por que", "recomienda", "conviene", "analiza", "compar", "riesgo", "diferencias entre"]
    data_signals = ["solo", "dame", "lista", "correos", "telefonos", "nombres", "ultimo", "kpi", "kpis"]

    has_analysis = any(signal in q for signal in analysis_signals)
    has_data = any(signal in q for signal in data_signals)

    if has_analysis and has_data:
        mode: ResponseMode = "hybrid"
    elif has_analysis:
        mode = "analysis"
    else:
        mode = "data"

    return QuestionIntent(
        mode=mode,
        wants_web=wants_web,
        asks_for_emails=asks_for_emails,
        asks_for_names=asks_for_names,
        asks_for_phones=asks_for_phones,
        asks_for_contact_directory=asks_for_contact_directory,
        asks_for_owner_load=asks_for_owner_load,
        asks_for_assigned_clients=asks_for_assigned_clients,
        asks_for_interactions_by_owner=asks_for_interactions_by_owner,
        asks_for_generic_relative_interactions=asks_for_generic_relative_interactions,
        asks_for_recent_activity_by_owner=asks_for_recent_activity_by_owner,
        asks_for_risks=asks_for_risks,
        asks_for_kpis=asks_for_kpis,
        asks_for_global_kpis=asks_for_global_kpis,
        asks_for_weekly_window=asks_for_weekly_window,
        asks_for_last_contact=asks_for_last_contact,
        asks_for_pending_commitments=asks_for_pending_commitments,
        asks_for_stale_contacts=asks_for_stale_contacts,
        asks_for_today_call_list=asks_for_today_call_list,
        asks_for_comparison=asks_for_comparison,
        asks_for_yesterday_contacts=asks_for_yesterday_contacts,
        asks_for_day_before_yesterday_contacts=asks_for_day_before_yesterday_contacts,
        asks_for_latest_contacted=asks_for_latest_contacted,
        asks_for_today_pending=asks_for_today_pending,
    )
