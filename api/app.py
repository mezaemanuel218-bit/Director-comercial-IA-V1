import os
import shutil
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field

from assistant_core.auth import AppUser, authenticate_user, get_user, session_secret_key
from assistant_core.config import BOOTSTRAP_WAREHOUSE_DB, PROJECT_ROOT, WAREHOUSE_DB
from assistant_core.documents import index_documents, indexed_documents_count
from assistant_core.history import ensure_history_schema, fetch_feedback, fetch_history, save_feedback, save_history
from assistant_core.reporting import available_owners, dashboard_metrics
from assistant_core.runtime_state import ensure_runtime_schema, get_runtime_value, set_runtime_value
from assistant_core.service import SalesAssistantService
from assistant_core.sync_runtime import (
    acquire_sync_lock,
    get_sync_snapshot,
    mark_sync_failure,
    mark_sync_success,
    refresh_is_stale,
    release_sync_lock,
)
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
AUTO_REFRESH_MAX_AGE_MINUTES = int(os.getenv("AUTO_REFRESH_MAX_AGE_MINUTES", "60"))
AUTO_REFRESH_LOOP_SECONDS = int(os.getenv("AUTO_REFRESH_LOOP_SECONDS", "300"))
AUTO_REFRESH_ENABLED = os.getenv("AUTO_REFRESH_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
INTERNAL_REFRESH_TOKEN = os.getenv("INTERNAL_REFRESH_TOKEN", "").strip()
_SYNC_THREAD_GUARD = threading.Lock()
_SYNC_LOOP_STARTED = False


def prepare_local_state() -> None:
    ensure_history_schema()
    ensure_runtime_schema()
    counts = warehouse_counts()
    if not any(counts.get(table, 0) for table in ("leads", "contacts", "notes", "calls", "tasks", "events", "interactions")):
        if BOOTSTRAP_WAREHOUSE_DB.exists():
            WAREHOUSE_DB.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BOOTSTRAP_WAREHOUSE_DB, WAREHOUSE_DB)
            counts = warehouse_counts()
    if not any(counts.get(table, 0) for table in ("leads", "contacts", "notes", "calls", "tasks", "events", "interactions")):
        build_warehouse()
        counts = warehouse_counts()
    if indexed_documents_count() == 0:
        index_documents()
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
            return parsed.replace(tzinfo=datetime.now().astimezone().tzinfo).astimezone(timezone.utc)
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


def _sync_modules() -> list[str]:
    return ["leads", "contacts", "notes", "calls", "tasks", "events"]


def _run_sync_pipeline(requested_by: str) -> None:
    if not acquire_sync_lock(requested_by):
        return
    try:
        set_runtime_value("last_refresh_attempt", requested_by)
        sync_zoho()
        build_warehouse()
        index_documents()
        mark_sync_success(requested_by)
    except Exception as exc:
        counts = warehouse_counts()
        if counts.get("leads", 0) > 0 or counts.get("contacts", 0) > 0:
            mark_sync_failure(str(exc), fallback_mode="snapshot_only")
        else:
            mark_sync_failure(str(exc), fallback_mode="error")
    finally:
        release_sync_lock()


def _start_sync_thread(requested_by: str) -> bool:
    with _SYNC_THREAD_GUARD:
        snapshot = get_sync_snapshot()
        if snapshot.sync_in_progress:
            return False
        thread = threading.Thread(target=_run_sync_pipeline, args=(requested_by,), daemon=True)
        thread.start()
        return True


def _sync_warning() -> str | None:
    snapshot = get_sync_snapshot()
    if snapshot.sync_in_progress:
        requester = snapshot.sync_requested_by or "otro proceso"
        return f"Hay una sincronizacion de Zoho en curso ({requester}). Se mantiene el ultimo snapshot disponible."
    if snapshot.refresh_mode == "snapshot_only":
        warning = "Usando snapshot local. La sincronizacion en vivo con Zoho no esta disponible en este momento."
        if snapshot.refresh_error:
            warning += f" Error: {snapshot.refresh_error}"
        return warning
    if snapshot.refresh_mode == "error" and snapshot.refresh_error:
        return f"La ultima sincronizacion fallo: {snapshot.refresh_error}"
    return None


def _build_refresh_response(status: str = "ok", warning: str | None = None) -> "RefreshResponse":
    counts = warehouse_counts()
    snapshot = get_sync_snapshot()
    return RefreshResponse(
        status=status,
        synced_modules=_sync_modules(),
        warehouse=counts,
        indexed_documents=indexed_documents_count(),
        last_refresh=snapshot.last_refresh,
        refresh_mode=snapshot.refresh_mode,
        warning=warning or _sync_warning(),
        cooldown_seconds=_refresh_cooldown_remaining_seconds(),
        sync_in_progress=snapshot.sync_in_progress,
        sync_requested_by=snapshot.sync_requested_by,
        sync_started_at=snapshot.sync_started_at,
    )


def _ensure_sync_loop() -> None:
    global _SYNC_LOOP_STARTED
    if _SYNC_LOOP_STARTED or not AUTO_REFRESH_ENABLED:
        return

    def loop() -> None:
        while True:
            try:
                if refresh_is_stale(AUTO_REFRESH_MAX_AGE_MINUTES):
                    _start_sync_thread("auto_scheduler")
            except Exception:
                pass
            time.sleep(max(60, AUTO_REFRESH_LOOP_SECONDS))

    with _SYNC_THREAD_GUARD:
        if _SYNC_LOOP_STARTED:
            return
        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        _SYNC_LOOP_STARTED = True


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
    history_id: int | None = None


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
    sync_in_progress: bool = False
    sync_requested_by: str | None = None
    sync_started_at: str | None = None


class HistoryItem(BaseModel):
    id: int
    username: str | None = None
    question: str
    answer: str
    mode: str | None = None
    sources: str | None = None
    used_web: int
    created_at: str


class FeedbackRequest(BaseModel):
    history_id: int = Field(..., ge=1)
    rating: str = Field(..., pattern="^(good|bad)$")
    correction: str | None = Field(default=None, max_length=4000)
    notes: str | None = Field(default=None, max_length=4000)


class FeedbackItem(BaseModel):
    id: int
    history_id: int
    username: str | None = None
    rating: str
    correction: str | None = None
    notes: str | None = None
    question: str
    answer: str
    mode: str | None = None
    sources: str | None = None
    created_at: str
    updated_at: str


class FeedbackResponse(BaseModel):
    status: str
    item: FeedbackItem


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
    sync_in_progress: bool = False
    sync_requested_by: str | None = None
    sync_started_at: str | None = None


class PriorityResponse(BaseModel):
    owner: str | None
    items: list[dict]


class UserResponse(BaseModel):
    username: str
    display_name: str
    role: str
    title: str
    crm_owner_name: str | None = None


def _to_user_response(user: AppUser) -> UserResponse:
    return UserResponse(
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        title=user.title,
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
    _ensure_sync_loop()
    if AUTO_REFRESH_ENABLED and refresh_is_stale(AUTO_REFRESH_MAX_AGE_MINUTES):
        _start_sync_thread("startup_auto")


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
        history_id = save_history(
            question=payload.question,
            answer=result.answer,
            mode=result.mode,
            sources=result.sources,
            used_web=result.used_web,
            username=user.username,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        mode=result.mode,
        sources=result.sources,
        used_web=result.used_web,
        answer=result.answer,
        history_id=history_id,
    )


@app.post("/refresh", response_model=RefreshResponse)
def refresh(user: AppUser = Depends(require_user)) -> RefreshResponse:
    snapshot = get_sync_snapshot()
    if snapshot.sync_in_progress:
        return _build_refresh_response(status="running")
    cooldown_seconds = _refresh_cooldown_remaining_seconds()
    if cooldown_seconds > 0:
        warning = (
            f"Para evitar saturar Zoho, la siguiente sincronizacion manual estara disponible "
            f"en aproximadamente {cooldown_seconds // 60 + (1 if cooldown_seconds % 60 else 0)} minuto(s). "
            "Mientras tanto se usa el ultimo snapshot cargado."
        )
        return _build_refresh_response(status="cooldown", warning=warning)

    started = _start_sync_thread(f"manual:{user.username}")
    if not started:
        return _build_refresh_response(status="running")
    return _build_refresh_response(
        status="queued",
        warning="Sincronizacion de Zoho iniciada en segundo plano. Puedes seguir usando la app mientras termina.",
    )


@app.post("/internal/refresh", response_model=RefreshResponse)
def internal_refresh(request: Request) -> RefreshResponse:
    if not INTERNAL_REFRESH_TOKEN:
        raise HTTPException(status_code=404, detail="No disponible.")
    if request.headers.get("x-refresh-token", "").strip() != INTERNAL_REFRESH_TOKEN:
        raise HTTPException(status_code=401, detail="Token invalido.")
    snapshot = get_sync_snapshot()
    if snapshot.sync_in_progress:
        return _build_refresh_response(status="running")
    started = _start_sync_thread("internal_auto")
    if not started:
        return _build_refresh_response(status="running")
    return _build_refresh_response(
        status="queued",
        warning="Sincronizacion interna iniciada en segundo plano.",
    )


@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = 20, user: AppUser = Depends(require_user)) -> list[HistoryItem]:
    try:
        items = fetch_history(limit=limit, username=user.username)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [HistoryItem(**item) for item in items]


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest, user: AppUser = Depends(require_user)) -> FeedbackResponse:
    try:
        feedback_id = save_feedback(
            history_id=payload.history_id,
            rating=payload.rating,
            username=user.username,
            correction=payload.correction,
            notes=payload.notes,
        )
        item = next(
            (row for row in fetch_feedback(limit=50, username=user.username) if row["id"] == feedback_id),
            None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not item:
        raise HTTPException(status_code=500, detail="No se pudo guardar el feedback.")
    return FeedbackResponse(status="ok", item=FeedbackItem(**item))


@app.get("/feedback", response_model=list[FeedbackItem])
def feedback_list(limit: int = 30, user: AppUser = Depends(require_user)) -> list[FeedbackItem]:
    try:
        items = fetch_feedback(limit=limit, username=user.username)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [FeedbackItem(**item) for item in items]


@app.get("/dashboard", response_model=DashboardResponse)
def dashboard(
    owner: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = None,
    user: AppUser = Depends(require_user),
) -> DashboardResponse:
    try:
        data = dashboard_metrics(owner=owner, date_from=date_from, date_to=date_to, status=status)
        cooldown_seconds = _refresh_cooldown_remaining_seconds()
        snapshot = get_sync_snapshot()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DashboardResponse(
        **data,
        last_refresh=snapshot.last_refresh,
        refresh_mode=snapshot.refresh_mode,
        warning=_sync_warning(),
        cooldown_seconds=cooldown_seconds,
        sync_in_progress=snapshot.sync_in_progress,
        sync_requested_by=snapshot.sync_requested_by,
        sync_started_at=snapshot.sync_started_at,
    )


@app.get("/owners", response_model=list[str])
def owners(user: AppUser = Depends(require_user)) -> list[str]:
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
