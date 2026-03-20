import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def nested_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def owner_fields(payload: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    owner = payload.get("Owner") or {}
    if not isinstance(owner, dict):
        return None, None, None
    return owner.get("id"), owner.get("name"), owner.get("email")


def entity_fields(payload: dict[str, Any], field_name: str) -> tuple[str | None, str | None]:
    entity = payload.get(field_name) or {}
    if not isinstance(entity, dict):
        return None, None
    return entity.get("id"), entity.get("name")


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    no_tags = HTML_TAG_RE.sub(" ", value)
    return WHITESPACE_RE.sub(" ", no_tags).strip()


def parse_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


def compact_text(*values: Any) -> str:
    parts = [str(value).strip() for value in values if value not in (None, "")]
    return " | ".join(parts)
