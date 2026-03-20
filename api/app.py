import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field

from assistant_core.auth import authenticate_user, get_user, session_secret_key
from assistant_core.config import PROJECT_ROOT
from assistant_core.documents import index_documents
from assistant_core.history import ensure_history_schema, fetch_history, save_history
from assistant_core.reporting import available_owners, dashboard_metrics
from assistant_core.runtime_state import get_runtime_value, set_runtime_value
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


def prepare_local_state() -> None:
    build_warehouse()
    index_documents()
    ensure_history_schema()
    _auto_refresh_if_empty()


def _auto_refresh_if_empty() -> None:
    counts = warehouse_counts()
    if counts.get("leads", 0) > 0 or counts.get("contacts", 0) > 0:
        return

    required = ["ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN"]
    if not all(os.getenv(key) for key in required):
        return

    try:
        sync_zoho()
        build_warehouse()
        index_documents()
        set_runtime_value("last_refresh", "startup_auto_refresh")
    except Exception:
        return


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


class PriorityResponse(BaseModel):
    owner: str | None
    items: list[dict]


class UserResponse(BaseModel):
    username: str
    display_name: str


def require_user(request: Request) -> UserResponse:
    session_user = request.session.get("username")
    user = get_user(session_user)
    if not user:
        raise HTTPException(status_code=401, detail="Sesión requerida.")
    return UserResponse(username=user.username, display_name=user.display_name)


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
    return UserResponse(username=user.username, display_name=user.display_name)


@app.post("/logout")
def logout(request: Request) -> dict[str, str]:
    request.session.clear()
    return {"status": "ok"}


@app.get("/me", response_model=UserResponse)
def me(user: UserResponse = Depends(require_user)) -> UserResponse:
    return user


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, user: UserResponse = Depends(require_user)) -> AskResponse:
    try:
        result = assistant_service.answer_question(payload.question)
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
def refresh(user: UserResponse = Depends(require_user)) -> RefreshResponse:
    try:
        sync_zoho()
        stats = build_warehouse()
        indexed_documents = index_documents()
        set_runtime_value("last_refresh", "manual_refresh")
        last_refresh = get_runtime_value("last_refresh")
    except Exception as exc:
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DashboardResponse(**data, last_refresh=last_refresh["updated_at"] if last_refresh else None)


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
