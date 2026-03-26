import unicodedata
from dataclasses import dataclass
from typing import Literal


ResponseMode = Literal["data", "analysis", "hybrid"]


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    ascii_text = ascii_text.replace("_", " ")
    return " ".join(ascii_text.lower().split())


@dataclass
class QuestionIntent:
    mode: ResponseMode
    wants_web: bool
    asks_for_emails: bool
    asks_for_names: bool
    asks_for_phones: bool
    asks_for_contact_directory: bool
    asks_for_client_brief: bool
    asks_for_owner_brief: bool
    asks_for_team_brief: bool
    asks_for_sales_draft: bool
    asks_for_action_plan: bool
    asks_for_sales_material: bool
    asks_for_formatted_output: bool
    asks_for_multi_step: bool
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
    asks_for_latest_note: bool


def classify_question(question: str) -> QuestionIntent:
    q = _normalize(question)

    asks_for_emails = any(token in q for token in ["correo", "correos", "mail", "email", "emails"])
    asks_for_names = any(token in q for token in ["nombre", "nombres", "persona", "personas", "contacto", "contactos"])
    asks_for_phones = any(token in q for token in ["telefono", "telefonos", "numero", "numeros", "celular", "movil"])
    asks_for_contact_directory = any(token in q for token in ["contacto", "contactos"]) and any(token in q for token in [" de ", " del ", ":"])
    asks_for_sales_draft = (
        any(token in q for token in ["correo", "mail", "email", "mensaje", "whatsapp", "propuesta"])
        and any(
            token in q
            for token in [
                "mandarle",
                "mandar",
                "redacta",
                "redactame",
                "escribe",
                "borrador",
                "draft",
                "usa todo lo que sabes",
                "fabrica",
                "hazme",
                "armame",
                "asunto y cuerpo",
                "correo para",
                "mensaje para",
            ]
        )
    ) or (
        "seguimiento para " in q
        and any(token in q for token in ["redacta", "fabrica", "escribe", "mensaje", "correo", "seguimiento"])
    )
    asks_for_action_plan = any(
        token in q
        for token in [
            "plan para hoy",
            "plan de trabajo",
            "plan de seguimiento",
            "plan de accion",
            "que haria hoy",
            "que harias hoy",
            "hoy, manana y esta semana",
            "hoy mañana y esta semana",
            "siguiente paso comercial",
            "siguiente paso",
            "que me recomiendas hacer hoy",
            "a quien debo contactar hoy",
            "a quien debo llamar hoy",
            "si solo pudiera hacer una accion hoy",
            "si solo pudiera hacer una acción hoy",
            "que oportunidades tengo mas calientes",
            "donde estoy dejando dinero en la mesa",
            "que clientes mios estan mas vivos",
            "que clientes mios ya se enfriaron",
            "si quisiera vender este mes",
            "que cliente de mi cartera esta mas cerca de avanzar",
            "que cliente de mi cartera ves mas frio pero recuperable",
            "que tres clientes debo atacar primero esta semana",
            "si fueras yo",
            "si fueras director comercial",
        ]
    )
    asks_for_sales_material = any(
        token in q
        for token in [
            "argumentos de venta",
            "speech de 30 segundos",
            "speech de 30 segundos",
            "bullets de valor",
            "beneficios de nuestros servicios",
            "beneficios de nuestros productos",
            "propuesta comercial breve",
            "mini agenda de reunion",
            "mini agenda de reunión",
            "preguntas de descubrimiento",
            "objeciones probables y como responderlas",
            "objeciones probables y cómo responderlas",
            "como responderlas",
            "como responderlas",
        ]
    )
    asks_for_formatted_output = any(
        token in q
        for token in [
            "como correo",
            "como whatsapp",
            "como plan de accion",
            "como plan de acción",
            "formato ejecutivo",
            "en bullets",
            "solo recomendacion y riesgos",
            "solo recomendación y riesgos",
            "conclusion y luego evidencia",
            "conclusión y luego evidencia",
        ]
    )
    asks_for_multi_step = any(
        token in q
        for token in [
            " y luego ",
            " luego ",
            "despues ",
            "después ",
            "detecta riesgo y redacta",
            "resume la cuenta, detecta riesgo y redacta",
            "dame resumen de ",
            "y proponme siguiente paso",
        ]
    )
    asks_for_client_brief = (
        any(token in q for token in [
            "todo lo que debo saber",
            "que debo saber",
            "resumen del cliente",
            "resumen del prospecto",
            "resumen comercial",
            "resumeme ",
            "estatus del cliente",
            "historial del cliente",
            "dame contexto de",
            "dame contexto comercial de",
            "que sabes de",
            "como vamos con",
            "que sigue con",
            "siguiente paso de",
            "objeciones de",
            "objeciones en",
            "resumen ejecutivo de",
            "si entro a una llamada en 5 minutos con",
            "si entro a una llamada con",
            "antes de una reunion con",
            "brief de cliente para antes de una reunion con",
        ])
        or (
            any(token in q for token in ["cliente", "prospecto", "lead", "cuenta", "empresa"])
            and any(token in q for token in ["resumen", "historial", "estatus", "contexto", "detalle", "detalles"])
        )
    )
    asks_for_owner_brief = any(
        token in q
        for token in [
            "mis contactos o leads",
            "mis contactos",
            "mis leads",
            "mi cartera",
            "resumen de mi cartera",
            "resumen del vendedor",
            "resumen comercial del vendedor",
            "kpi mio",
            "mis kpis",
            "mis logros",
            "mi actividad comercial",
            "mis pendientes",
            "clientes calientes",
            "clientes frios",
            "mis notas",
            "mis oportunidades",
        ]
    )
    asks_for_team_brief = any(
        token in q
        for token in [
            "resumen del equipo",
            "resumen comercial del equipo",
            "kpi global",
            "kpis globales",
            "logros del equipo",
            "actividad del equipo",
            "panorama del equipo",
        ]
    )

    asks_for_owner_load = (
        any(token in q for token in ["quien tiene mas clientes asignados", "carga por vendedor", "quien tiene mas prospectos asignados"])
        or (
            any(token in q for token in ["carga", "asignados"])
            and any(token in q for token in ["vendedor", "propietario", "agente", "owner", "owners", "por vendedor"])
        )
        or (
            "cuantos clientes" in q
            and any(token in q for token in ["por vendedor", "cada vendedor", "vendedor", "propietario", "agente"])
        )
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
    asks_for_recent_activity_by_owner = (
        (any(token in q for token in ["actividad reciente", "mas actividad reciente"]) and any(token in q for token in ["quien", "vendedor", "de "]))
        or ("que actividad esta realizando" in q and any(token in q for token in ["crm", "vendedor", "eduardo", "pablo", "emmanuel"]))
    )
    asks_for_risks = any(token in q for token in ["riesgo", "riesgos", "objecion", "objeciones", "bloqueador", "bloqueadores", "decision maker", "vale la pena seguir insistiendo"])
    asks_for_kpis = "kpi" in q or "kpis" in q or ("metric" in q and any(token in q for token in ["nota", "cliente"]))
    asks_for_global_kpis = asks_for_kpis and any(token in q for token in ["global", "todos", "vendedores", "equipo"])
    asks_for_weekly_window = "semana" in q
    asks_for_last_contact = any(token in q for token in ["ultimo contacto", "ayer", "antier", "anteayer"])
    asks_for_pending_commitments = any(token in q for token in ["compromiso", "pendiente", "tarea"])
    asks_for_stale_contacts = any(token in q for token in ["30 dias", "sin contacto"])
    asks_for_today_call_list = (
        "a quien debo llamar hoy" in q
        or "hoy y por que" in q
        or "a quien debo contactar hoy" in q
        or "a quien me recomiendas llamar hoy" in q
        or "a quien me recomiendas contactar hoy" in q
        or "a quien debo darle seguimiento hoy" in q
        or ("hoy" in q and any(token in q for token in ["llamar", "llame", "contactar", "contacte", "seguimiento"]) and any(token in q for token in ["debo", "recomiendas", "recomiendas", "conviene"]))
        or ("en base a mis notas" in q and any(token in q for token in ["llamar", "contactar", "seguimiento"]))
    )
    asks_for_comparison = any(token in q for token in ["compara", "comparativa", " vs ", "diferencia entre", "diferencias entre"])
    wants_web = any(token in q for token in ["web", "internet"])
    asks_for_yesterday_contacts = "ayer" in q and any(token in q for token in ["hable", "llame", "contacte"])
    asks_for_day_before_yesterday_contacts = any(token in q for token in ["antier", "anteayer"]) and any(token in q for token in ["hable", "llame", "contacte"])
    asks_for_latest_contacted = any(token in q for token in ["ultimo cliente", "cliente que se contacto a lo ultimo"])
    asks_for_today_pending = "hoy" in q and any(token in q for token in ["pendiente", "pendientes", "compromiso", "compromisos", "tarea", "tareas"])
    asks_for_latest_note = any(token in q for token in ["ultima nota", "última nota", "nota agregada", "nota mas reciente", "nota más reciente"])

    analysis_signals = [
        "plan",
        "estrategia",
        "por que",
        "recomienda",
        "conviene",
        "analiza",
        "compar",
        "riesgo",
        "diferencias entre",
        "en base a mis notas",
        "que haria",
        "que harias",
        "donde estoy dejando dinero",
        "oportunidades calientes",
        "siguiente paso",
        "argumentos de venta",
        "propuesta comercial",
    ]
    data_signals = ["solo", "dame", "lista", "correos", "telefonos", "nombres", "ultimo", "kpi", "kpis", "asunto y cuerpo"]

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
        asks_for_client_brief=asks_for_client_brief,
        asks_for_owner_brief=asks_for_owner_brief,
        asks_for_team_brief=asks_for_team_brief,
        asks_for_sales_draft=asks_for_sales_draft,
        asks_for_action_plan=asks_for_action_plan,
        asks_for_sales_material=asks_for_sales_material,
        asks_for_formatted_output=asks_for_formatted_output,
        asks_for_multi_step=asks_for_multi_step,
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
        asks_for_latest_note=asks_for_latest_note,
    )
