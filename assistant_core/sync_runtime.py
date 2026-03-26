from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from assistant_core.runtime_state import get_runtime_value, set_runtime_value


SYNC_LOCK_TIMEOUT_MINUTES = 45


@dataclass
class SyncSnapshot:
    last_refresh: str | None
    refresh_mode: str | None
    refresh_error: str | None
    sync_in_progress: bool
    sync_requested_by: str | None
    sync_started_at: str | None


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


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_sync_snapshot() -> SyncSnapshot:
    last_refresh = get_runtime_value("last_refresh")
    refresh_status = get_runtime_value("last_refresh_status")
    refresh_error = get_runtime_value("last_refresh_error")
    sync_in_progress = get_runtime_value("sync_in_progress")
    sync_requested_by = get_runtime_value("sync_requested_by")
    sync_started_at = get_runtime_value("sync_started_at")
    return SyncSnapshot(
        last_refresh=last_refresh["updated_at"] if last_refresh else None,
        refresh_mode=refresh_status["value"] if refresh_status else None,
        refresh_error=refresh_error["value"] if refresh_error and refresh_error["value"] else None,
        sync_in_progress=bool(sync_in_progress and sync_in_progress["value"] == "1"),
        sync_requested_by=sync_requested_by["value"] if sync_requested_by else None,
        sync_started_at=sync_started_at["updated_at"] if sync_started_at else None,
    )


def sync_lock_stale(timeout_minutes: int = SYNC_LOCK_TIMEOUT_MINUTES) -> bool:
    snapshot = get_sync_snapshot()
    if not snapshot.sync_in_progress:
        return False
    started_at = _parse_runtime_timestamp(snapshot.sync_started_at)
    if not started_at:
        return True
    return utcnow() - started_at > timedelta(minutes=timeout_minutes)


def acquire_sync_lock(requested_by: str, timeout_minutes: int = SYNC_LOCK_TIMEOUT_MINUTES) -> bool:
    snapshot = get_sync_snapshot()
    if snapshot.sync_in_progress and not sync_lock_stale(timeout_minutes):
        return False
    set_runtime_value("sync_in_progress", "1")
    set_runtime_value("sync_requested_by", requested_by)
    set_runtime_value("sync_started_at", requested_by)
    return True


def release_sync_lock() -> None:
    set_runtime_value("sync_in_progress", "0")
    set_runtime_value("sync_requested_by", "")
    set_runtime_value("sync_started_at", "")


def mark_sync_success(mode: str) -> None:
    set_runtime_value("last_refresh", mode)
    set_runtime_value("last_refresh_status", "ok")
    set_runtime_value("last_refresh_error", "")


def mark_sync_failure(error: str, fallback_mode: str = "snapshot_only") -> None:
    set_runtime_value("last_refresh_status", fallback_mode)
    set_runtime_value("last_refresh_error", error)


def refresh_is_stale(max_age_minutes: int) -> bool:
    snapshot = get_sync_snapshot()
    if not snapshot.last_refresh:
        return True
    last_refresh_at = _parse_runtime_timestamp(snapshot.last_refresh)
    if not last_refresh_at:
        return True
    if snapshot.refresh_mode in {None, "snapshot_only", "startup_snapshot"}:
        return True
    return utcnow() - last_refresh_at > timedelta(minutes=max_age_minutes)
