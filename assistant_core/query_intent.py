from dataclasses import dataclass
from typing import Literal


ResponseMode = Literal["data", "analysis", "hybrid"]


@dataclass
class QuestionIntent:
    mode: ResponseMode
    wants_web: bool
    asks_for_emails: bool
    asks_for_names: bool
    asks_for_phones: bool
    asks_for_owner_load: bool
    asks_for_assigned_clients: bool
    asks_for_interactions_by_owner: bool
    asks_for_recent_activity_by_owner: bool
    asks_for_risks: bool
    asks_for_kpis: bool
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
    q = question.lower()

    asks_for_emails = "correo" in q or "mail" in q or "email" in q
    asks_for_names = "nombre" in q or "nombres" in q or "persona" in q or "personas" in q or "contacto" in q or "contactos" in q
    asks_for_phones = "telefono" in q or "teléfono" in q or "numero" in q or "número" in q
    asks_for_owner_load = (
        ("cuantos clientes" in q or "cuántos clientes" in q or "asignados" in q or "carga" in q)
        and ("vendedor" in q or "propietario" in q or "agente" in q or "clientes" in q)
    ) or (
        "quien tiene mas clientes asignados" in q
        or "quién tiene más clientes asignados" in q
        or "carga por vendedor" in q
        or "quien tiene mas prospectos asignados" in q
        or "quién tiene más prospectos asignados" in q
    )
    asks_for_assigned_clients = (
        "mis clientes" in q
        or "quienes son mis clientes" in q
        or "quiénes son mis clientes" in q
        or "clientes de " in q
        or "clientes asignados de " in q
        or "prospectos de " in q
        or "prospectos asignados de " in q
    )
    asks_for_interactions_by_owner = "interacciones" in q and ("vendedor" in q or "agente" in q or "propietario" in q or "cada" in q)
    asks_for_recent_activity_by_owner = ("actividad reciente" in q or "mas actividad reciente" in q or "más actividad reciente" in q) and ("quien" in q or "quién" in q or "vendedor" in q)
    asks_for_risks = "riesgo" in q or "riesgos" in q
    asks_for_kpis = "kpi" in q or "kpis" in q or ("metric" in q and ("nota" in q or "cliente" in q))
    asks_for_last_contact = "ultimo contacto" in q or "último contacto" in q or "ayer" in q or "antier" in q or "anteayer" in q
    asks_for_pending_commitments = "compromiso" in q or "pendiente" in q or "tarea" in q
    asks_for_stale_contacts = "30 dias" in q or "30 días" in q or "sin contacto" in q
    asks_for_today_call_list = "a quien debo llamar hoy" in q or "a quién debo llamar hoy" in q or "hoy y por que" in q
    asks_for_comparison = "compara" in q or "comparativa" in q or "vs" in q or "diferencia entre" in q
    wants_web = "web" in q or "internet" in q
    asks_for_yesterday_contacts = (
        ("ayer" in q)
        and ("hable" in q or "hablé" in q or "llame" in q or "llamé" in q or "contacte" in q or "contacté" in q)
    )
    asks_for_day_before_yesterday_contacts = (
        ("antier" in q or "anteayer" in q)
        and ("hable" in q or "hablé" in q or "llame" in q or "llamé" in q or "contacte" in q or "contacté" in q)
    )
    asks_for_latest_contacted = "ultimo cliente" in q or "último cliente" in q or "cliente que se contacto a lo ultimo" in q or "cliente que se contactó a lo último" in q
    asks_for_today_pending = ("hoy" in q) and ("pendiente" in q or "compromiso" in q or "tarea" in q)

    analysis_signals = [
        "plan",
        "estrategia",
        "por que",
        "por qué",
        "recomienda",
        "conviene",
        "analiza",
        "compar",
        "riesgo",
        "riesgos",
    ]
    data_signals = [
        "solo",
        "dame",
        "lista",
        "correos",
        "telefonos",
        "teléfonos",
        "nombres",
        "ultimo",
        "último",
    ]

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
        asks_for_owner_load=asks_for_owner_load,
        asks_for_assigned_clients=asks_for_assigned_clients,
        asks_for_interactions_by_owner=asks_for_interactions_by_owner,
        asks_for_recent_activity_by_owner=asks_for_recent_activity_by_owner,
        asks_for_risks=asks_for_risks,
        asks_for_kpis=asks_for_kpis,
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
