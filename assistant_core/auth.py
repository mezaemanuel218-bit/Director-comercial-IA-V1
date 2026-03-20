import hashlib
import hmac
import os
from dataclasses import dataclass


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AppUser:
    username: str
    display_name: str
    role: str
    crm_owner_name: str | None
    password_hash: str


DEFAULT_PASSWORD_HASH = _hash_password("Flotimatics2026")

APP_USERS = {
    "evaldez": AppUser(
        username="evaldez",
        display_name="Eduardo Valdez",
        role="ceo",
        crm_owner_name="Eduardo Valdez",
        password_hash=DEFAULT_PASSWORD_HASH,
    ),
    "pmelin": AppUser(
        username="pmelin",
        display_name="Pablo Melin",
        role="seller",
        crm_owner_name="Pablo Melin Dorador",
        password_hash=DEFAULT_PASSWORD_HASH,
    ),
    "emeza": AppUser(
        username="emeza",
        display_name="Emmanuel Meza",
        role="seller",
        crm_owner_name="Jesus Emmanuel Meza Guzm\u00e1n",
        password_hash=DEFAULT_PASSWORD_HASH,
    ),
}


def session_secret_key() -> str:
    return os.getenv("APP_SECRET_KEY", "flotimatics-director-comercial-ia-dev-secret")


def authenticate_user(username: str, password: str) -> AppUser | None:
    normalized = username.strip().lower()
    user = APP_USERS.get(normalized)
    if not user:
        return None

    candidate_hash = _hash_password(password)
    if hmac.compare_digest(candidate_hash, user.password_hash):
        return user
    return None


def get_user(username: str | None) -> AppUser | None:
    if not username:
        return None
    return APP_USERS.get(username.strip().lower())
