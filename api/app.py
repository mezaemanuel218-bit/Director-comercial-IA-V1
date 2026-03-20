import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field

from assistant_core.auth import AppUser, authenticate_user, get_user, session_secret_key
from assistant_core.config import PROJECT_ROOT
from assistant_core.documents import index_documents
from assistant_core.history import ensure_history_schema, fetch_history, save_history
from assistant_core.reporting import available_owners, dashboard_metrics
from assistant_core.runtime_state import ensure_runtime_schema, get_runtime_value, set_runtime_value
from assistant_core.service import SalesAssistantService
from assistant_core.warehouse import build_warehouse, warehouse_counts
from scripts.sync_zoho import run as sync_zoho


app = FastAPI(
    title="Director Comercial IA API",
    version="0.1.0",
    description="API base para el asistente comercial de Flotimatics.",
)

app.add_middleware(
    SessionMiddleware,
    secret_key=session_secret_key(),
    same_site="lax",
    https_only=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant_service = SalesAssistantService()
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
ensure_history_schema()
ensure_runtime_schema()
ZOHO_REFRESH_COOLDOWN_MINUTES = 10


def prepare_local_state() -> None:
    build_warehouse()
    index_documents()
    ensure_history_schema()
    ensure_runtime_schema()
    counts = warehouse_counts()
    has_snapshot = counts.get("leads", 0) > 0 or counts.get("contacts", 0) > 0
    existing_refresh = get_runtime_value("last_refresh")
    existing_status = get_runtime_value("last_refresh_status")
    if has_snapshot and not existing_refresh:
        set_runtime_value("last_refresh", "startup_snapshot")
    if has_snapshot and not existing_status:
        set_runtime_value("last_refresh_status", "snapshot_only")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_runtime_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _refresh_cooldown_remaining_seconds() -> int:
    last_attempt = get_runtime_value("last_refresh_attempt")
    if not last_attempt:
        return 0
    attempt_time = _parse_runtime_timestamp(last_attempt["updated_at"])
    if not attempt_time:
        return 0
    elapsed = _utcnow() - attempt_time
    cooldown = timedelta(minutes=ZOHO_REFRESH_COOLDOWN_MINUTES)
    remaining = cooldown - elapsed
    return max(0, int(remaining.total_seconds()))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Pregunta del usuario")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=2)
    password: str = Field(..., min_length=6)


class AskResponse(BaseModel):
    mode: str
    sources: list[str]
    used_web: bool
    answer: str


class HealthResponse(BaseModel):
    status: str
    service: str


class RefreshResponse(BaseModel):
    status: str
    synced_modules: list[str]
    warehouse: dict[str, int]
    indexed_documents: int
    last_refresh: str | None = None
    refresh_mode: str | None = None
    warning: str | None = None
    cooldown_seconds: int = 0


class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    mode: str | None = None
    sources: str | None = None
    used_web: int
    created_at: str


class DashboardResponse(BaseModel):
    filters: dict[str, str | None]
    total_interactions: int
    by_type: list[dict]
    owner_load: list[dict]
    pending_tasks: list[dict]
    stale_contacts: list[dict]
    recent_activity: list[dict]
    last_refresh: str | None = None
    refresh_mode: str | None = None
    warning: str | None = None
    cooldown_seconds: int = 0


class PriorityResponse(BaseModel):
    owner: str | None
    items: list[dict]


class UserResponse(BaseModel):
    username: str
    display_name: str
    role: str
    crm_owner_name: str | None = None


def _to_user_response(user: AppUser) -> UserResponse:
    return UserResponse(
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        crm_owner_name=user.crm_owner_name,
    )


def require_user(request: Request) -> AppUser:
    session_user = request.session.get("username")
    user = get_user(session_user)
    if not user:
        raise HTTPException(status_code=401, detail="Sesión requerida.")
    return user


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.on_event("startup")
def startup_event() -> None:
    prepare_local_state()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="director-comercial-ia")


@app.post("/login", response_model=UserResponse)
def login(payload: LoginRequest, request: Request) -> UserResponse:
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Usuario o contraseña inválidos.")
    request.session["username"] = user.username
    return _to_user_response(user)


@app.post("/logout")
def logout(request: Request) -> dict[str, str]:
    request.session.clear()
    return {"status": "ok"}


@app.get("/me", response_model=UserResponse)
def me(user: AppUser = Depends(require_user)) -> UserResponse:
    return _to_user_response(user)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, user: AppUser = Depends(require_user)) -> AskResponse:
    try:
        result = assistant_service.answer_question(payload.question, user=user)
        save_history(
            question=payload.question,
            answer=result.answer,
            mode=result.mode,
            sources=result.sources,
            used_web=result.used_web,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        mode=result.mode,
        sources=result.sources,
        used_web=result.used_web,
        answer=result.answer,
    )


@app.post("/refresh", response_model=RefreshResponse)
def refresh(user: AppUser = Depends(require_user)) -> RefreshResponse:
    cooldown_seconds = _refresh_cooldown_remaining_seconds()
    if cooldown_seconds > 0:
        counts = warehouse_counts()
        last_refresh = get_runtime_value("last_refresh")
        refresh_status = get_runtime_value("last_refresh_status")
        return RefreshResponse(
            status="ok",
            synced_modules=["leads", "contacts", "notes", "calls", "tasks", "events"],
            warehouse=counts,
            indexed_documents=0,
            last_refresh=last_refresh["updated_at"] if last_refresh else None,
            refresh_mode=refresh_status["value"] if refresh_status else "snapshot_only",
            warning=(
                f"Para evitar saturar Zoho, la siguiente sincronizacion manual estara disponible "
                f"en aproximadamente {cooldown_seconds // 60 + (1 if cooldown_seconds % 60 else 0)} minuto(s). "
                "Mientras tanto se usa el ultimo snapshot cargado."
            ),
            cooldown_seconds=cooldown_seconds,
        )

    warning = None
    try:
        set_runtime_value("last_refresh_attempt", "manual_attempt")
        sync_zoho()
        stats = build_warehouse()
        indexed_documents = index_documents()
        set_runtime_value("last_refresh", "manual_refresh")
        set_runtime_value("last_refresh_status", "ok")
        set_runtime_value("last_refresh_error", "")
        last_refresh = get_runtime_value("last_refresh")
        refresh_status = get_runtime_value("last_refresh_status")
    except Exception as exc:
        counts = warehouse_counts()
        if counts.get("leads", 0) > 0 or counts.get("contacts", 0) > 0:
            set_runtime_value("last_refresh_status", "snapshot_only")
            set_runtime_value("last_refresh_error", str(exc))
            last_refresh = get_runtime_value("last_refresh")
            refresh_status = get_runtime_value("last_refresh_status")
            warning = "No se pudo sincronizar Zoho en vivo. Se mantiene la informacion del ultimo snapshot cargado."
        return RefreshResponse(
            status="ok",
            synced_modules=["leads", "contacts", "notes", "calls", "tasks", "events"],
            warehouse=counts,
            indexed_documents=0,
            last_refresh=last_refresh["updated_at"] if last_refresh else None,
            refresh_mode=refresh_status["value"] if refresh_status else "snapshot_only",
            warning=warning,
            cooldown_seconds=ZOHO_REFRESH_COOLDOWN_MINUTES * 60,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RefreshResponse(
        status="ok",
        synced_modules=["leads", "contacts", "notes", "calls", "tasks", "events"],
        warehouse={
            "leads": stats.leads,
            "contacts": stats.contacts,
            "notes": stats.notes,
            "calls": stats.calls,
            "tasks": stats.tasks,
            "events": stats.events,
            "interactions": stats.interactions,
        },
        indexed_documents=indexed_documents,
        last_refresh=last_refresh["updated_at"] if last_refresh else None,
        refresh_mode=refresh_status["value"] if refresh_status else "ok",
        warning=warning,
        cooldown_seconds=ZOHO_REFRESH_COOLDOWN_MINUTES * 60,
    )


@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = 20, user: UserResponse = Depends(require_user)) -> list[HistoryItem]:
    try:
        items = fetch_history(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [HistoryItem(**item) for item in items]


@app.get("/dashboard", response_model=DashboardResponse)
def dashboard(
    owner: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = None,
    user: UserResponse = Depends(require_user),
) -> DashboardResponse:
    try:
        data = dashboard_metrics(owner=owner, date_from=date_from, date_to=date_to, status=status)
        last_refresh = get_runtime_value("last_refresh")
        refresh_status = get_runtime_value("last_refresh_status")
        refresh_error = get_runtime_value("last_refresh_error")
        cooldown_seconds = _refresh_cooldown_remaining_seconds()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    warning = None
    if refresh_status and refresh_status["value"] == "snapshot_only":
        warning = "Usando snapshot local. La sincronizacion en vivo con Zoho no esta disponible en este momento."
        if refresh_error and refresh_error["value"]:
            warning += f" Error: {refresh_error['value']}"
    return DashboardResponse(
        **data,
        last_refresh=last_refresh["updated_at"] if last_refresh else None,
        refresh_mode=refresh_status["value"] if refresh_status else None,
        warning=warning,
        cooldown_seconds=cooldown_seconds,
    )


@app.get("/owners", response_model=list[str])
def owners(user: UserResponse = Depends(require_user)) -> list[str]:
    try:
        return available_owners()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/priorities", response_model=PriorityResponse)
def priorities(owner: str | None = None, limit: int = 12, user: UserResponse = Depends(require_user)) -> PriorityResponse:
    try:
        items = assistant_service.get_priority_followups(owner=owner, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PriorityResponse(owner=owner, items=items)
