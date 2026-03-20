from dataclasses import dataclass
from typing import Literal


ResponseMode = Literal["data", "analysis", "hybrid"]


@dataclass
class QuestionIntent:
    mode: ResponseMode
    wants_web: bool
    asks_for_emails: bool
    asks_for_phones: bool
    asks_for_owner_load: bool
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
    asks_for_phones = "telefono" in q or "teléfono" in q or "numero" in q or "número" in q
    asks_for_owner_load = (
        ("cuantos clientes" in q or "cuántos clientes" in q or "asignados" in q)
        and ("vendedor" in q or "propietario" in q or "agente" in q)
    )
    asks_for_last_contact = "ultimo contacto" in q or "último contacto" in q or "ayer" in q or "antier" in q or "anteayer" in q
    asks_for_pending_commitments = "compromiso" in q or "pendiente" in q or "tarea" in q
    asks_for_stale_contacts = "30 dias" in q or "30 días" in q or "sin contacto" in q
    asks_for_today_call_list = "a quien debo llamar hoy" in q or "a quién debo llamar hoy" in q or "hoy y por que" in q
    asks_for_comparison = "compara" in q or "comparativa" in q or "vs" in q
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
        asks_for_phones=asks_for_phones,
        asks_for_owner_load=asks_for_owner_load,
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
